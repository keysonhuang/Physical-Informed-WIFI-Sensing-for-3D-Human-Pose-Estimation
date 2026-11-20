from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pywt
from scipy.io import loadmat


def _fill_non_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=-1, keepdims=True)
    total = np.where(finite, values, 0.0).sum(axis=-1, keepdims=True)
    mean = total / np.maximum(count, 1)
    return np.where(finite, values, mean)


def sanitize_pair_phase(pair_phase: np.ndarray, subcarrier_spacing_hz: float) -> np.ndarray:
    """Unwrap each link phase and remove its least-squares frequency slope."""
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


def _as_link_packets(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 4:
        if values.shape[0] * values.shape[1] != 9:
            raise ValueError(f"{name} 4D input must contain nine spatial links, got {values.shape}.")
        values = values.reshape(9, values.shape[-2], values.shape[-1])
    elif values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [links, subcarriers, packets], got {values.shape}.")
    return values


def features_from_link_amplitude_phase(
    amplitude: np.ndarray,
    phase: np.ndarray,
    subcarrier_spacing_hz: float = 625000.0,
    sanitize_phase: bool = True,
) -> np.ndarray:
    """Convert CSI from spatial links into [links * subcarriers, (amplitude, phase)]."""
    amplitude = _as_link_packets(amplitude, "amplitude").astype(np.float64, copy=False)
    phase = _as_link_packets(phase, "phase").astype(np.float64, copy=False)
    if amplitude.shape != phase.shape:
        raise ValueError(f"Amplitude and phase shapes differ: {amplitude.shape} vs {phase.shape}.")
    amplitude = np.maximum(_fill_non_finite(amplitude), 0.0)
    phase = _fill_non_finite(phase)
    if sanitize_phase:
        phase = sanitize_pair_phase(phase, subcarrier_spacing_hz)
    packet_scale = amplitude.mean(axis=(0, 1), keepdims=True)
    amplitude = amplitude / np.maximum(packet_scale, 1e-8)
    amplitude_feature = amplitude.mean(axis=-1)
    phase_feature = np.angle(np.exp(1j * phase).mean(axis=-1)) / np.pi
    features = np.stack((amplitude_feature, phase_feature), axis=-1)
    return np.nan_to_num(features.reshape(-1, 2).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_csi_arrays(
    amplitude: np.ndarray,
    phase: np.ndarray,
    subcarrier_spacing_hz: float = 312500.0,
) -> np.ndarray:
    """MM-Fi preprocessing: cyclic conjugate products across three receive antennas."""
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
        raise ValueError(f"Expected three receive antennas for MM-Fi, got {amplitude.shape[0]}.")
    amplitude = np.maximum(_fill_non_finite(amplitude), 0.0)
    phase = _fill_non_finite(phase)
    channel = amplitude * np.exp(1j * phase)
    conjugate_pairs = channel * np.conjugate(np.roll(channel, shift=-1, axis=0))
    pair_amplitude = np.sqrt(np.maximum(np.abs(conjugate_pairs), 0.0))
    pair_phase = sanitize_pair_phase(np.angle(conjugate_pairs), subcarrier_spacing_hz)
    packet_scale = pair_amplitude.mean(axis=(0, 1), keepdims=True)
    pair_amplitude = pair_amplitude / np.maximum(packet_scale, 1e-8)
    features = np.stack(
        (pair_amplitude.mean(axis=-1), np.angle(np.exp(1j * pair_phase).mean(axis=-1)) / np.pi),
        axis=-1,
    )
    return np.nan_to_num(features.reshape(-1, 2).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_person_wifi_complex(csi: np.ndarray, subcarrier_spacing_hz: float) -> np.ndarray:
    """Phase sanitization for the public Person-in-WiFi-3D 3x3xSxT CSI tensor."""
    csi = np.asarray(csi)
    if csi.ndim != 4 or csi.shape[0:2] != (3, 3):
        raise ValueError(f"Expected complex CSI shaped [3 RX, 3 TX, subcarriers, packets], got {csi.shape}.")
    num_subcarriers, num_packets = csi.shape[2:]
    phase = np.zeros((3, 3, num_subcarriers, num_packets), dtype=np.float64)
    subcarrier_index = np.arange(num_subcarriers, dtype=np.float64)
    design_frequency = np.tile(2.0 * np.pi * float(subcarrier_spacing_hz) * subcarrier_index, 3)
    design = np.stack((design_frequency, np.ones_like(design_frequency)), axis=1)
    for rx in range(3):
        for packet in range(num_packets):
            first = np.unwrap(np.angle(csi[rx, 0, :, packet]))
            second = np.unwrap(first + np.angle(csi[rx, 1, :, packet] * np.conjugate(csi[rx, 0, :, packet])))
            third = np.unwrap(second + np.angle(csi[rx, 2, :, packet] * np.conjugate(csi[rx, 1, :, packet])))
            stacked = np.concatenate((first, second, third))
            rho, beta = np.linalg.lstsq(design, -stacked, rcond=None)[0]
            correction = 2.0 * np.pi * float(subcarrier_spacing_hz) * subcarrier_index * rho + beta
            phase[rx, 0, :, packet] = first + correction
            phase[rx, 1, :, packet] = second + correction
            phase[rx, 2, :, packet] = third + correction
    return np.abs(csi) * np.exp(1j * phase)


@lru_cache(maxsize=8192)
def load_person_wifi_raw_csi(path: str | Path, subcarrier_spacing_hz: float = 625000.0) -> np.ndarray:
    """Read the official HDF5 MAT layout and return 270x2 frame features."""
    with h5py.File(path, "r") as handle:
        if "csi_out" not in handle:
            raise KeyError(f"{path} must contain the csi_out HDF5 group.")
        group = handle["csi_out"]
        if "real" not in group or "imag" not in group:
            raise KeyError(f"{path}: csi_out must contain real and imag arrays.")
        raw = np.asarray(group["real"]) + 1j * np.asarray(group["imag"])
    if raw.ndim != 4:
        raise ValueError(f"Expected four-dimensional csi_out array, got {raw.shape}.")
    # Official files store [packets, subcarriers, TX, RX].
    csi = raw.transpose(3, 2, 1, 0).astype(np.complex128, copy=False)
    sanitized = _sanitize_person_wifi_complex(csi, subcarrier_spacing_hz)
    amplitude = np.abs(sanitized)
    coefficients = pywt.wavedec(amplitude, pywt.Wavelet("db11"), mode="symmetric", axis=-1)
    amplitude = pywt.waverec(coefficients, pywt.Wavelet("db11"), mode="symmetric", axis=-1)
    amplitude = amplitude[..., :sanitized.shape[-1]]
    phase = np.angle(sanitized)
    amplitude = amplitude.reshape(9, sanitized.shape[2], sanitized.shape[3])
    phase = phase.reshape(9, sanitized.shape[2], sanitized.shape[3])
    return features_from_link_amplitude_phase(amplitude, phase, subcarrier_spacing_hz, sanitize_phase=False)


@lru_cache(maxsize=8192)
def load_person_wifi_preprocessed(path: str | Path) -> np.ndarray:
    """Read official csi_ap .npy files shaped [3, 180, 20], or canonical [270, 2]."""
    data = np.asarray(np.load(path))
    if data.shape == (270, 2):
        return data.astype(np.float32, copy=True)
    if data.ndim != 3 or data.shape[0] != 3 or data.shape[1] != 180:
        raise ValueError(f"Expected csi_ap shape [3, 180, packets] or [270, 2], got {data.shape}.")
    amplitude = data[:, :90, :].reshape(3, 3, 30, data.shape[-1]).reshape(9, 30, data.shape[-1])
    phase = data[:, 90:, :].reshape(3, 3, 30, data.shape[-1]).reshape(9, 30, data.shape[-1])
    return features_from_link_amplitude_phase(amplitude, phase, sanitize_phase=False)


@lru_cache(maxsize=8192)
def _load_sanitized_csi_cached(path: str, subcarrier_spacing_hz: float) -> np.ndarray:
    mat = loadmat(path)
    if "CSIamp" not in mat or "CSIphase" not in mat:
        raise KeyError(f"{path} must contain CSIamp and CSIphase arrays.")
    return preprocess_csi_arrays(mat["CSIamp"], mat["CSIphase"], subcarrier_spacing_hz)


def load_sanitized_csi(path: str | Path, subcarrier_spacing_hz: float = 312500.0) -> np.ndarray:
    path = str(Path(path).resolve())
    return _load_sanitized_csi_cached(path, float(subcarrier_spacing_hz)).copy()


def clear_csi_cache() -> None:
    _load_sanitized_csi_cached.cache_clear()
