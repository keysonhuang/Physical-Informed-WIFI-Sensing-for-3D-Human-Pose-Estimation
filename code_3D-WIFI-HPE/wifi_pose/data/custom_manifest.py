from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .csi import features_from_link_amplitude_phase, load_person_wifi_raw_csi


def _load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        with np.load(path) as archive:
            selected = key or archive.files[0]
            if selected not in archive:
                raise KeyError(f"{path} does not contain key '{selected}'. Available: {archive.files}")
            return np.asarray(archive[selected])
    if suffix == ".mat":
        try:
            data = loadmat(path)
            selected = key or "pose"
            if selected not in data:
                raise KeyError(f"{path} does not contain key '{selected}'.")
            return np.asarray(data[selected])
        except (NotImplementedError, ValueError):
            with h5py.File(path, "r") as handle:
                selected = key or "pose"
                if selected not in handle:
                    raise KeyError(f"{path} does not contain HDF5 key '{selected}'.")
                return np.asarray(handle[selected])
    raise ValueError(f"Unsupported array file extension: {path.suffix}")


def _load_custom_csi(path: Path, config: dict[str, Any]) -> np.ndarray:
    csi_format = str(config.get("csi_format", "amplitude_phase")).lower()
    spacing = float(config.get("signal", {}).get("subcarrier_spacing_hz", 625000.0))
    if csi_format == "person_in_wifi_hdf5":
        return load_person_wifi_raw_csi(path, spacing)
    if csi_format == "features":
        features = _load_array(path, config.get("features_key", "features"))
        if features.ndim != 2 or features.shape[-1] != 2:
            raise ValueError(f"Precomputed CSI features must be [links * subcarriers, 2], got {features.shape}.")
        return features.astype(np.float32, copy=False)
    if csi_format == "amplitude_phase":
        amplitude = _load_array(path, config.get("amplitude_key", "CSIamp"))
        phase = _load_array(path, config.get("phase_key", "CSIphase"))
        return features_from_link_amplitude_phase(
            amplitude, phase, spacing, sanitize_phase=bool(config.get("sanitize_phase", True))
        )
    if csi_format == "complex":
        csi = _load_array(path, config.get("complex_key", "csi"))
        if not np.iscomplexobj(csi):
            raise ValueError(f"Complex CSI format selected, but {path} is not a complex array.")
        if csi.ndim == 4:
            if csi.shape[0] * csi.shape[1] != 9:
                raise ValueError(f"Complex CSI must have nine spatial links, got {csi.shape}.")
            csi = csi.reshape(9, csi.shape[-2], csi.shape[-1])
        if csi.ndim == 2:
            csi = csi[..., None]
        if csi.ndim != 3:
            raise ValueError(f"Complex CSI must be [links, subcarriers, packets], got {csi.shape}.")
        return features_from_link_amplitude_phase(np.abs(csi), np.angle(csi), spacing)
    raise ValueError("custom.csi_format must be features, amplitude_phase, complex, or person_in_wifi_hdf5.")


class ManifestPoseDataset(Dataset):
    """Dataset adapter driven by a CSV manifest; raw schemas are configured, never guessed."""

    def __init__(self, data_root: str | Path, config: dict[str, Any], split: str):
        self.root = Path(data_root).expanduser().resolve()
        self.config = config
        self.split = "train" if split == "train" else "val"
        manifest_path = Path(config.get("manifest", "manifest.csv"))
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Custom dataset manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"Custom dataset manifest is empty: {manifest_path}")
        required = {"csi_path", "pose_path", "split"}
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        if config.get("require_action_environment_labels", False):
            missing_labels = {"action_id", "environment_id"} - set(rows[0])
            if missing_labels:
                raise ValueError(f"Manifest requires action_id and environment_id columns: {sorted(missing_labels)}")

        self.records: list[dict[str, Any]] = []
        for row in rows:
            row_split = row["split"].strip().lower()
            normalized_split = "train" if row_split in {"train", "training"} else "val" if row_split in {"val", "valid", "validation", "test"} else row_split
            if normalized_split != self.split:
                continue
            csi_path, pose_path = Path(row["csi_path"]), Path(row["pose_path"])
            if not csi_path.is_absolute():
                csi_path = self.root / csi_path
            if not pose_path.is_absolute():
                pose_path = self.root / pose_path
            if not csi_path.is_file() or not pose_path.is_file():
                raise FileNotFoundError(f"Manifest row references missing files: {csi_path}, {pose_path}")
            pose = _load_array(pose_path, config.get("pose_key", "pose"))
            if pose.ndim == 3:
                person_index = config.get("person_index")
                if person_index is None:
                    if pose.shape[0] != 1:
                        raise ValueError(
                            f"{pose_path} contains {pose.shape[0]} people; configure person_index or export one person."
                        )
                    pose = pose[0]
                else:
                    pose = pose[int(person_index)]
            if pose.ndim != 2 or pose.shape[-1] != 3:
                raise ValueError(f"Pose annotations must be [joints, 3], got {pose.shape} in {pose_path}")
            sample_id = row.get("sample_id") or csi_path.stem
            sequence_id = row.get("sequence_id") or sample_id
            try:
                frame_index = int(row.get("frame_index") or 0)
                action_id = int(row["action_id"]) if row.get("action_id", "").strip() else int(config.get("default_action_id", 0))
                environment_id = int(row["environment_id"]) if row.get("environment_id", "").strip() else int(config.get("default_environment_id", -1))
            except ValueError as exc:
                raise ValueError(f"Invalid numeric metadata for sample {sample_id}") from exc
            if config.get("require_action_environment_labels", False) and (
                not row.get("action_id", "").strip() or not row.get("environment_id", "").strip()
            ):
                raise ValueError(f"Sample {sample_id} must have action_id and environment_id labels.")
            self.records.append({
                "sample_id": sample_id, "sequence_id": sequence_id, "frame_index": frame_index,
                "csi_path": csi_path, "pose": pose.astype(np.float32, copy=False),
                "action_id": action_id, "environment_id": environment_id,
            })
        if not self.records:
            raise RuntimeError(f"No {self.split} samples found in {manifest_path}")
        self.sequences: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            self.sequences.setdefault(record["sequence_id"], []).append(index)
        for indices in self.sequences.values():
            indices.sort(key=lambda idx: self.records[idx]["frame_index"])
        self.frame_index = [
            (sequence_id, position)
            for sequence_id, indices in self.sequences.items()
            for position in range(len(indices))
        ]
        self.window_size = int(config.get("window_size", 10))
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sequence_id, position = self.frame_index[index]
        indices = self.sequences[sequence_id]
        center = self.records[indices[position]]
        left = (self.window_size - 1) // 2
        window_indices = [
            indices[min(max(position - left + offset, 0), len(indices) - 1)]
            for offset in range(self.window_size)
        ]
        window = np.stack([
            _load_custom_csi(self.records[idx]["csi_path"], self.config) for idx in window_indices
        ])
        static = window.mean(axis=0)
        current = window[left]
        return {
            "dynamic": torch.from_numpy((current - static).astype(np.float32)),
            "static": torch.from_numpy(static.astype(np.float32)),
            "pose": torch.from_numpy(center["pose"].copy()),
            "action": int(center["action_id"]),
            "environment": int(center["environment_id"]),
        }
