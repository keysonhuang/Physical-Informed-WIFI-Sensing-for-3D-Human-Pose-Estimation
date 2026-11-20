from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _fill_non_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=-1, keepdims=True)
    total = np.where(finite, values, 0.0).sum(axis=-1, keepdims=True)
    mean = total / np.maximum(count, 1)
    return np.where(finite, values, mean)


def sanitize_pair_phase(
    pair_phase: np.ndarray,
    subcarrier_spacing_hz: float,
) -> np.ndarray:
    """Unwrap each antenna-pair phase and remove its least-squares frequency slope."""
    phase = np.unwrap(np.asarray(pair_phase, dtype=np.float64), axis=1)
    num_subcarriers = phase.shape[1]
    frequencies = (np.arange(num_subcarriers, dtype=np.float64) - (num_subcarriers - 1) / 2.0)
    frequencies *= float(subcarrier_spacing_hz)
    centered = frequencies - frequencies.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return phase
    phase_centered = phase - phase.mean(axis=1, keepdims=True)
    slope = np.sum(phase_centered * centered[None, :, None], axis=1, keepdims=True) / denominator
    return phase - slope * centered[None, :, None]


def preprocess_csi_arrays(
    amplitude: np.ndarray,
    phase: np.ndarray,
    subcarrier_spacing_hz: float = 312500.0,
) -> np.ndarray:
    """Convert one pose-aligned CSI record into an [antenna-pair × subcarrier, 2] tensor."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    if amplitude.ndim == 2:
        amplitude = amplitude[..., None]
    if phase.ndim == 2:
        phase = phase[..., None]
    if amplitude.ndim != 3 or phase.ndim != 3:
        raise ValueError("CSIamp and CSIphase must have shape [antennas, subcarriers, packets].")
    if amplitude.shape != phase.shape:
        raise ValueError(f"Amplitude and phase shapes differ: {amplitude.shape} vs {phase.shape}.")
    if amplitude.shape[0] != 3:
        raise ValueError(f"Expected three receive antennas, got {amplitude.shape[0]}.")

    amplitude = np.maximum(_fill_non_finite(amplitude), 0.0)
    phase = _fill_non_finite(phase)
    channel = amplitude * np.exp(1j * phase)

    next_antenna = np.roll(channel, shift=-1, axis=0)
    conjugate_pairs = channel * np.conjugate(next_antenna)

    pair_amplitude = np.sqrt(np.maximum(np.abs(conjugate_pairs), 0.0))
    pair_phase = sanitize_pair_phase(np.angle(conjugate_pairs), subcarrier_spacing_hz)

    packet_scale = pair_amplitude.mean(axis=(0, 1), keepdims=True)
    pair_amplitude = pair_amplitude / np.maximum(packet_scale, 1e-8)

    amplitude_feature = pair_amplitude.mean(axis=-1)
    phase_feature = pair_phase.mean(axis=-1) / np.pi

    features = np.stack((amplitude_feature, phase_feature), axis=-1)
    features = features.reshape(-1, 2).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


@lru_cache(maxsize=8192)
def _load_sanitized_csi_cached(
    path: str,
    subcarrier_spacing_hz: float,
) -> np.ndarray:
    mat = loadmat(path)
    if "CSIamp" not in mat or "CSIphase" not in mat:
        raise KeyError(f"{path} must contain CSIamp and CSIphase arrays.")
    return preprocess_csi_arrays(mat["CSIamp"], mat["CSIphase"], subcarrier_spacing_hz)


def load_sanitized_csi(path: str | Path, subcarrier_spacing_hz: float = 312500.0) -> np.ndarray:
    path = str(Path(path).resolve())
    return _load_sanitized_csi_cached(path, float(subcarrier_spacing_hz)).copy()


def clear_csi_cache() -> None:
    _load_sanitized_csi_cached.cache_clear()
