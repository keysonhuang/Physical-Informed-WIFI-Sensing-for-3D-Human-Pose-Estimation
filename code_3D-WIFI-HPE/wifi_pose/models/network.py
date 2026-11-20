from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .pgma import PhysicsGuidedMultipathEncoder


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = F.relu(self.norm1(self.conv1(features)), inplace=True)
        features = self.dropout(features)
        features = self.norm2(self.conv2(features))
        return F.relu(features + residual, inplace=True)


class EnvironmentAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        return self.network(static_features)


class WiFiPoseEstimator(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        signal_config = data_config.get("signal", {})

        self.num_antennas = int(signal_config.get("num_receive_antennas", 3))
        self.num_subcarriers = int(signal_config.get("num_subcarriers", 114))
        self.num_tokens = self.num_antennas * self.num_subcarriers
        self.num_actions = int(model_config.get("num_actions", 27))
        self.num_joints = int(model_config.get("num_joints", 17))
        self.d_model = int(model_config.get("d_model", 64))
        self.action_embedding_dim = int(model_config.get("action_embedding_dim", 128))
        self.num_environments = int(model_config.get("num_environments", 4))
        self.dynamic_kernel_size = int(model_config.get("dynamic_kernel_size", 5))
        split_mode = str(data_config.get("split", "")).lower()
        if "source_environment_ids" in data_config:
            self.source_environment_ids = [int(value) for value in data_config["source_environment_ids"]]
        elif split_mode == "cross_environment":
            self.source_environment_ids = [int(str(value)[1:]) - 1 for value in data_config.get("train_environments", ["E01", "E02", "E03"])]
        else:
            self.source_environment_ids = list(range(self.num_environments))
        if split_mode == "cross_environment":
            self.adapter_environment_ids = list(self.source_environment_ids)
        else:
            self.adapter_environment_ids = list(range(self.num_environments))
        self.adapter_index = {environment_id: index for index, environment_id in enumerate(self.adapter_environment_ids)}

        self.encoder = PhysicsGuidedMultipathEncoder(
            num_antennas=self.num_antennas,
            num_subcarriers=self.num_subcarriers,
            center_frequency_hz=float(signal_config.get("center_frequency_hz", 5.0e9)),
            subcarrier_spacing_hz=float(signal_config.get("subcarrier_spacing_hz", 312500.0)),
            antenna_spacing_m=signal_config.get("antenna_spacing_m"),
            sigma_antenna_m=signal_config.get("sigma_antenna_m"),
            sigma_frequency_hz=float(signal_config.get("sigma_frequency_hz", 6.0e6)),
            d_model=self.d_model,
            num_heads=int(model_config.get("num_heads", 8)),
            num_layers=int(model_config.get("encoder_layers", 2)),
            feedforward_dim=int(model_config.get("feedforward_dim", 128)),
            num_bands=int(model_config.get("num_frequency_bands", 3)),
            dropout=float(model_config.get("dropout", 0.1)),
        )

        self.action_classifier = nn.Sequential(
            nn.Linear(self.d_model, self.action_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.action_embedding_dim, self.num_actions),
        )
        self.action_embedding = nn.Linear(self.num_actions, self.action_embedding_dim, bias=False)

        kernel_elements = self.d_model * self.d_model * self.dynamic_kernel_size * self.dynamic_kernel_size
        self.kernel_hypernetwork = nn.Linear(self.action_embedding_dim, kernel_elements, bias=False)
        nn.init.normal_(self.kernel_hypernetwork.weight, mean=0.0, std=1e-3)

        self.coarse_resnet = nn.Sequential(ResidualBlock2D(self.d_model, dropout=0.0))
        self.coarse_key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.coarse_value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.action_query = nn.Linear(self.action_embedding_dim, self.d_model, bias=False)
        self.fusion_gate = nn.Linear(self.d_model * 2 + self.action_embedding_dim, self.d_model)

        adapter_input_dim = self.num_tokens * 2
        adapter_hidden_dim = int(model_config.get("environment_hidden_dim", 128))
        self.environment_adapters = nn.ModuleList(
            [EnvironmentAdapter(adapter_input_dim, adapter_hidden_dim, self.d_model) for _ in self.adapter_environment_ids]
        )

        joint_input_dim = self.d_model + self.action_embedding_dim
        self.joint_gates = nn.ModuleList(
            [nn.Linear(joint_input_dim, self.d_model) for _ in range(self.num_joints)]
        )
        self.joint_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 3),
                )
                for _ in range(self.num_joints)
            ]
        )

    def _dynamic_convolution(self, tokens: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        height = self.num_antennas
        width = self.num_subcarriers
        spatial = tokens.transpose(1, 2).reshape(batch_size, self.d_model, height, width)
        kernel = self.kernel_hypernetwork(action_embedding).view(
            batch_size,
            self.d_model,
            self.d_model,
            self.dynamic_kernel_size,
            self.dynamic_kernel_size,
        )
        grouped_input = spatial.reshape(1, batch_size * self.d_model, height, width)
        grouped_kernel = kernel.reshape(
            batch_size * self.d_model,
            self.d_model,
            self.dynamic_kernel_size,
            self.dynamic_kernel_size,
        )
        convolved = F.conv2d(
            grouped_input,
            grouped_kernel,
            padding=self.dynamic_kernel_size // 2,
            groups=batch_size,
        )
        return convolved.view(batch_size, self.d_model, height, width)

    def _environment_residual(
        self,
        static_csi: torch.Tensor,
        environment_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = static_csi.shape[0]
        flattened = static_csi.reshape(batch_size, -1)
        if environment_ids is None:
            environment_ids = torch.full(
                (batch_size,), -1, dtype=torch.long, device=static_csi.device
            )
        else:
            environment_ids = environment_ids.to(device=static_csi.device, dtype=torch.long)

        residual = static_csi.new_zeros((batch_size, self.d_model))
        known_mask = torch.zeros(batch_size, dtype=torch.bool, device=static_csi.device)
        for environment_id, adapter_index in self.adapter_index.items():
            mask = environment_ids.eq(environment_id)
            if mask.any():
                residual[mask] = self.environment_adapters[adapter_index](flattened[mask])
                known_mask |= mask

        unknown_mask = ~known_mask
        if unknown_mask.any():
            source_ids = [value for value in self.source_environment_ids if value in self.adapter_index]
            if not source_ids:
                source_ids = list(self.adapter_environment_ids)
            source_outputs = [
                self.environment_adapters[self.adapter_index[value]](flattened[unknown_mask])
                for value in source_ids
            ]
            residual[unknown_mask] = torch.stack(source_outputs, dim=0).mean(dim=0)
        return residual

    def forward(
        self,
        dynamic_csi: torch.Tensor,
        static_csi: torch.Tensor,
        environment_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.encoder(dynamic_csi)
        dynamic_feature = tokens.mean(dim=1)

        action_logits = self.action_classifier(dynamic_feature)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        action_embedding = self.action_embedding(action_probabilities)

        coarse_map = self._dynamic_convolution(tokens, action_embedding)
        coarse_map = self.coarse_resnet(F.relu(coarse_map, inplace=True))
        coarse_tokens = coarse_map.flatten(2).transpose(1, 2)
        coarse_feature = coarse_tokens.mean(dim=1)

        query = self.action_query(action_embedding).unsqueeze(1)
        keys = self.coarse_key(coarse_tokens)
        values = self.coarse_value(coarse_tokens)
        attention = torch.softmax(torch.bmm(query, keys.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        cross_feature = torch.bmm(attention, values).squeeze(1)

        gate_input = torch.cat((coarse_feature, action_embedding, cross_feature), dim=-1)
        fusion_gate = torch.sigmoid(self.fusion_gate(gate_input))
        reference_feature = fusion_gate * cross_feature + (1.0 - fusion_gate) * coarse_feature

        adapted_feature = reference_feature + self._environment_residual(static_csi, environment_ids)
        joint_context = torch.cat((adapted_feature, action_embedding), dim=-1)
        joint_predictions = []
        for gate_layer, decoder in zip(self.joint_gates, self.joint_decoders):
            joint_weight = torch.sigmoid(gate_layer(joint_context))
            joint_feature = joint_weight * adapted_feature
            joint_predictions.append(decoder(joint_feature))
        pose = torch.stack(joint_predictions, dim=1)

        return {
            "pose": pose,
            "action_logits": action_logits,
            "dynamic_feature": dynamic_feature,
            "reference_feature": reference_feature,
            "adapted_feature": adapted_feature,
        }


def build_model(config: dict) -> WiFiPoseEstimator:
    return WiFiPoseEstimator(config)
