from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .csi import load_person_wifi_preprocessed, load_person_wifi_raw_csi


def _read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split list: {path}")
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sequence_and_frame(sample_name: str) -> tuple[str, int]:
    parts = sample_name.split("_")
    if len(parts) >= 3:
        try:
            return "_".join(parts[:2]), int(parts[2])
        except ValueError:
            pass
    return sample_name, 0


def _load_single_person_pose(path: Path) -> np.ndarray | None:
    pose = np.asarray(np.load(path))
    if pose.ndim == 3:
        if pose.shape[0] != 1:
            return None
        pose = pose[0]
    if pose.ndim != 2 or pose.shape[-1] != 3:
        raise ValueError(f"Expected one 3D pose [joints, 3] in {path}, got {pose.shape}.")
    return pose.astype(np.float32, copy=False)


class PersonInWiFi3DPoseDataset(Dataset):
    """Single-person adapter for the official Person-in-WiFi-3D train/test layout."""

    def __init__(self, data_root: str | Path, config: dict[str, Any], split: str):
        self.data_root = Path(data_root).expanduser().resolve()
        self.config = config
        self.split = "train" if split == "train" else "test"
        self.window_size = int(config.get("window_size", 10))
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        self.subcarrier_spacing_hz = float(config.get("signal", {}).get("subcarrier_spacing_hz", 625000.0))
        split_dir = self.data_root / ("train_data" if self.split == "train" else "test_data")
        names_file = split_dir / ("train_data_list.txt" if self.split == "train" else "test_data_list.txt")
        names = _read_names(names_file)
        action_map = config.get("action_map", {})
        environment_map = config.get("environment_map", {})
        default_action = int(config.get("default_action_id", 0))
        default_environment = int(config.get("default_environment_id", -1))
        self.records: list[dict[str, Any]] = []

        for name in names:
            pose_path = split_dir / "keypoint" / f"{name}.npy"
            if not pose_path.is_file():
                raise FileNotFoundError(f"Missing keypoint annotation: {pose_path}")
            pose = _load_single_person_pose(pose_path)
            if pose is None:
                continue
            csi_ap_path = split_dir / "csi_ap" / f"{name}.npy"
            csi_path = split_dir / "csi" / f"{name}.mat"
            if csi_ap_path.is_file():
                source_path, source_format = csi_ap_path, "preprocessed"
            elif csi_path.is_file():
                source_path, source_format = csi_path, "raw"
            else:
                raise FileNotFoundError(f"Missing CSI for sample {name}: expected {csi_ap_path} or {csi_path}")

            sequence_id, frame_index = _sequence_and_frame(name)
            action_id = int(action_map.get(name, action_map.get(sequence_id, default_action)))
            environment_id = int(environment_map.get(name, environment_map.get(sequence_id, default_environment)))
            self.records.append({
                "name": name,
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "csi_path": source_path,
                "csi_format": source_format,
                "pose": pose,
                "action_id": action_id,
                "environment_id": environment_id,
            })

        if not self.records:
            raise RuntimeError(
                f"No single-person samples found in {split_dir}. Check the official split list and keypoint files."
            )

        self.sequences: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            self.sequences.setdefault(record["sequence_id"], []).append(index)
        for indices in self.sequences.values():
            indices.sort(key=lambda idx: self.records[idx]["frame_index"])
        self.frame_index = [(sequence_id, position) for sequence_id, indices in self.sequences.items()
                            for position in range(len(indices))]

    def __len__(self) -> int:
        return len(self.frame_index)

    def _load_csi(self, record: dict[str, Any]) -> np.ndarray:
        if record["csi_format"] == "preprocessed":
            return load_person_wifi_preprocessed(record["csi_path"])
        return load_person_wifi_raw_csi(record["csi_path"], self.subcarrier_spacing_hz)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sequence_id, position = self.frame_index[index]
        indices = self.sequences[sequence_id]
        center_record = self.records[indices[position]]
        left = (self.window_size - 1) // 2
        window_indices = [
            indices[min(max(position - left + offset, 0), len(indices) - 1)]
            for offset in range(self.window_size)
        ]
        window = np.stack([self._load_csi(self.records[idx]) for idx in window_indices], axis=0)
        static = window.mean(axis=0)
        current = window[left]
        return {
            "dynamic": torch.from_numpy((current - static).astype(np.float32)),
            "static": torch.from_numpy(static.astype(np.float32)),
            "pose": torch.from_numpy(center_record["pose"].copy()),
            "action": int(center_record["action_id"]),
            "environment": int(center_record["environment_id"]),
        }
