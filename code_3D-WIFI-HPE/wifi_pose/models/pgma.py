from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PGMALayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        num_bands: int,
        physical_prior: torch.Tensor,
        band_ids: torch.Tensor,
        prior_residual: nn.Parameter,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_bands = num_bands
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.band_weight = nn.Linear(d_model, num_bands)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm_attention = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_feedforward = nn.LayerNorm(d_model)
        prior = physical_prior.clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("prior_logits", torch.log(prior) - torch.log1p(-prior))
        self.register_buffer("band_ids", band_ids.long())
        self.prior_residual = prior_residual

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tokens.shape
        qkv = self.qkv(tokens).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + F.logsigmoid(self.prior_logits + self.prior_residual).unsqueeze(0).unsqueeze(0)

        band_outputs = []
        for band in range(self.num_bands):
            key_mask = self.band_ids.eq(band)
            query_mask = key_mask.to(dtype=tokens.dtype).view(1, 1, sequence_length, 1)
            band_scores = scores.masked_fill(~key_mask.view(1, 1, 1, sequence_length), -1e4)
            attention = self.attention_dropout(torch.softmax(band_scores, dim=-1))
            band_outputs.append(torch.matmul(attention, value) * query_mask)

        weights = torch.softmax(self.band_weight(tokens.mean(dim=1)), dim=-1)
        mixed = torch.zeros_like(band_outputs[0])
        for band, output in enumerate(band_outputs):
            mixed = mixed + output * weights[:, band].view(batch_size, 1, 1, 1)
        mixed = mixed.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        tokens = self.norm_attention(tokens + self.output_dropout(self.output_projection(mixed)))
        return self.norm_feedforward(tokens + self.feedforward(tokens))


class PhysicsGuidedMultipathEncoder(nn.Module):
    """PGMA encoder for an arbitrary configured number of spatial CSI links."""

    def __init__(
        self,
        num_antennas: int = 3,
        num_subcarriers: int = 114,
        center_frequency_hz: float = 5.0e9,
        subcarrier_spacing_hz: float = 312500.0,
        antenna_spacing_m: float | None = None,
        sigma_antenna_m: float | None = None,
        channel_positions_m: list | None = None,
        sigma_frequency_hz: float = 6.0e6,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 128,
        num_bands: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_antennas < 1:
            raise ValueError("num_antennas (spatial CSI channels) must be positive.")
        if num_subcarriers < num_bands:
            raise ValueError("num_subcarriers must be at least num_bands.")
        speed_of_light = 299792458.0
        wavelength = speed_of_light / center_frequency_hz
        if antenna_spacing_m is None:
            antenna_spacing_m = 0.5 * wavelength
        if sigma_antenna_m is None:
            sigma_antenna_m = max(float(antenna_spacing_m), 1e-4)

        if channel_positions_m is not None:
            coordinates = torch.as_tensor(channel_positions_m, dtype=torch.float32)
            if coordinates.shape[0] != num_antennas:
                raise ValueError("signal.channel_positions_m must have one position per spatial CSI channel.")
            if coordinates.ndim == 1:
                coordinates = coordinates[:, None]
            if coordinates.ndim != 2 or coordinates.shape[1] not in (1, 2, 3):
                raise ValueError("channel_positions_m must be [channels] or [channels, 2/3].")
            channel_coordinate = torch.linalg.vector_norm(coordinates, dim=-1)
            channel_distances = torch.cdist(coordinates, coordinates)
        elif num_antennas == 3:
            # Preserve the MM-Fi three cyclic antenna-pair geometry.
            channel_coordinate = torch.tensor(
                [antenna_spacing_m, antenna_spacing_m, 2.0 * antenna_spacing_m], dtype=torch.float32
            )
            channel_distances = torch.abs(channel_coordinate[:, None] - channel_coordinate[None, :])
        else:
            # Do not invent physical coordinates for distributed links. Frequency coherence
            # remains active; supply channel_positions_m to enable a spatial coherence prior.
            channel_coordinate = torch.linspace(0.0, 1.0, num_antennas)
            channel_distances = torch.zeros((num_antennas, num_antennas), dtype=torch.float32)

        subcarrier_offsets = (
            torch.arange(num_subcarriers, dtype=torch.float32) - (num_subcarriers - 1) / 2.0
        ) * subcarrier_spacing_hz
        frequencies = center_frequency_hz + subcarrier_offsets
        channel_positions = channel_coordinate.repeat_interleave(num_subcarriers)
        token_frequencies = frequencies.repeat(num_antennas)
        wavelengths = speed_of_light / token_frequencies
        phase_position = 2.0 * math.pi * channel_positions / wavelength
        phase_frequency = 2.0 * math.pi * token_frequencies / token_frequencies.max()
        positional_encoding = torch.stack(
            (torch.sin(phase_position), torch.cos(phase_position),
             torch.sin(phase_frequency), torch.cos(phase_frequency)), dim=-1
        )

        spatial_distance = channel_distances.repeat_interleave(num_subcarriers, dim=0).repeat_interleave(
            num_subcarriers, dim=1
        )
        frequency_difference = token_frequencies[:, None] - token_frequencies[None, :]
        prior = torch.exp(
            -(spatial_distance.square() / (2.0 * sigma_antenna_m**2))
            -(frequency_difference.square() / (2.0 * sigma_frequency_hz**2))
        )
        subcarrier_band_ids = torch.div(
            torch.arange(num_subcarriers) * num_bands,
            num_subcarriers,
            rounding_mode="floor",
        ).clamp(max=num_bands - 1)
        band_ids = subcarrier_band_ids.repeat(num_antennas)

        self.num_tokens = num_antennas * num_subcarriers
        self.position_projection = nn.Linear(4, 2, bias=False)
        self.input_projection = nn.Linear(2, d_model)
        prior = prior.clamp(1e-4, 1.0 - 1e-4)
        shared_prior_residual = nn.Parameter(torch.zeros_like(prior))
        self.layers = nn.ModuleList([
            PGMALayer(d_model, num_heads, feedforward_dim, num_bands, prior, band_ids,
                      shared_prior_residual, dropout)
            for _ in range(num_layers)
        ])
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, dynamic_csi: torch.Tensor) -> torch.Tensor:
        if dynamic_csi.ndim != 3 or dynamic_csi.shape[1:] != (self.num_tokens, 2):
            raise ValueError(
                f"Expected dynamic CSI with shape [batch, {self.num_tokens}, 2], got {tuple(dynamic_csi.shape)}."
            )
        tokens = self.input_projection(
            dynamic_csi + self.position_projection(self.positional_encoding).unsqueeze(0)
        )
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens
