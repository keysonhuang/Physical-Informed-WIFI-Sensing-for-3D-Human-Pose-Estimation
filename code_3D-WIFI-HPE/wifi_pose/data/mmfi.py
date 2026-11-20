from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .csi import load_sanitized_csi


ALL_ACTIONS = tuple(f"A{i:02d}" for i in range(1, 28))
DAILY_ACTIONS = ("A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A27")
REHABILITATION_ACTIONS = ("A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A15", "A16", "A24", "A25", "A26")


def environment_id(scene: str) -> int:
    match = re.fullmatch(r"E(\d{2})", scene)
    if not match:
        raise ValueError(f"Invalid environment name: {scene}")
    return int(match.group(1)) - 1


def _natural_frame_key(path: Path) -> int:
    match = re.search(r"frame(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _action_ids(protocol: str) -> set[str]:
    protocol = protocol.upper()
    if protocol == "P1":
        return set(DAILY_ACTIONS)
    if protocol == "P2":
        return set(REHABILITATION_ACTIONS)
    if protocol == "P3":
        return set(ALL_ACTIONS)
    raise ValueError("data.protocol must be P1, P2, or P3.")


class MMFiPoseDataset(Dataset):
    """Frame-level dataset with a centered temporal CSI window."""

    def __init__(self, data_root: str | Path, config: dict[str, Any], split: str):
        self.data_root = Path(data_root).expanduser().resolve()
        self.config = config
        self.split = split
        self.window_size = int(config.get("window_size", 10))
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        self.signal_config = config.get("signal", {})
        self.subcarrier_spacing_hz = float(self.signal_config.get("subcarrier_spacing_hz", 312500.0))
        self.allowed_actions = _action_ids(config.get("protocol", "P3"))
        self.records = self._discover_records()
        self.records = self._select_split(self.records)
        self.frame_index: list[tuple[int, int]] = []
        for record_index, record in enumerate(self.records):
            self.frame_index.extend((record_index, frame_index) for frame_index in range(record["length"]))
        if not self.frame_index:
            raise RuntimeError(f"No valid {split} frames found under {self.data_root}.")

    def _discover_records(self) -> list[dict[str, Any]]:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_root}")
        records: list[dict[str, Any]] = []
        for scene_dir in sorted(self.data_root.glob("E[0-9][0-9]")):
            if not scene_dir.is_dir():
                continue
            for subject_dir in sorted(scene_dir.glob("S[0-9][0-9]")):
                for action_dir in sorted(subject_dir.glob("A[0-9][0-9]")):
                    action = action_dir.name
                    if action not in self.allowed_actions:
                        continue
                    csi_dir = action_dir / "wifi-csi"
                    pose_path = action_dir / "ground_truth.npy"
                    if not csi_dir.is_dir() or not pose_path.is_file():
                        continue
                    frame_paths = sorted(csi_dir.glob("frame*.mat"), key=_natural_frame_key)
                    if not frame_paths:
                        continue
                    poses = np.load(pose_path, mmap_mode="r")
                    if poses.ndim != 3 or poses.shape[1:] != (17, 3):
                        raise ValueError(f"Expected 17x3 pose annotations in {pose_path}, got {poses.shape}.")
                    length = min(len(frame_paths), int(poses.shape[0]))
                    if length == 0:
                        continue
                    records.append({
                        "scene": scene_dir.name,
                        "environment": environment_id(scene_dir.name),
                        "subject": subject_dir.name,
                        "action": action,
                        "action_id": int(action[1:]) - 1,
                        "frame_paths": frame_paths[:length],
                        "pose_path": pose_path,
                        "length": length,
                        "key": f"{scene_dir.name}/{subject_dir.name}/{action}",
                    })
        if not records:
            raise RuntimeError(f"No valid MM-Fi recordings found under {self.data_root}.")
        return records

    def _select_split(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        split_mode = str(self.config.get("split", "cross_environment")).lower()
        if split_mode == "random":
            keys = [record["key"] for record in records]
            random.Random(int(self.config.get("random_seed", 42))).shuffle(keys)
            cutoff = max(1, min(len(keys) - 1, int(round(len(keys) * float(self.config.get("train_ratio", 0.8))))))
            selected = set(keys[:cutoff] if self.split == "train" else keys[cutoff:])
            return [record for record in records if record["key"] in selected]

        if split_mode == "cross_subject":
            subject_key = "train_subjects" if self.split == "train" else "test_subjects"
            subjects = self.config.get(subject_key)
            if not subjects:
                raise ValueError(f"data.{subject_key} must be configured for cross_subject split.")
            allowed = set(subjects)
            return [record for record in records if record["subject"] in allowed]

        if split_mode == "cross_environment":
            environment_key = "train_environments" if self.split == "train" else "test_environments"
            environments = self.config.get(environment_key)
            if not environments:
                raise ValueError(f"data.{environment_key} must be configured for cross_environment split.")
            allowed = set(environments)
            return [record for record in records if record["scene"] in allowed]

        raise ValueError("data.split must be random, cross_subject, or cross_environment.")

    def __len__(self) -> int:
        return len(self.frame_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record_index, frame_index = self.frame_index[index]
        record = self.records[record_index]
        paths = record["frame_paths"]
        left_context = (self.window_size - 1) // 2
        center_index = left_context
        start = frame_index - left_context
        offsets = range(self.window_size)
        window_indices = [min(max(start + offset, 0), record["length"] - 1) for offset in offsets]
        window = np.stack([
            load_sanitized_csi(paths[window_index], self.subcarrier_spacing_hz)
            for window_index in window_indices
        ], axis=0)
        static = window.mean(axis=0)
        current = window[center_index]
        dynamic = current - static
        pose_array = np.load(record["pose_path"], mmap_mode="r")[frame_index]

        return {
            "dynamic": torch.from_numpy(dynamic.astype(np.float32)),
            "static": torch.from_numpy(static.astype(np.float32)),
            "pose": torch.from_numpy(np.array(pose_array, dtype=np.float32, copy=True)),
            "action": int(record["action_id"]),
            "environment": int(record["environment"]),
        }


def build_dataloaders(config: dict[str, Any], data_root: str | Path | None = None) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    root = data_root if data_root is not None else data_config["root"]
    train_dataset = MMFiPoseDataset(root, data_config, "train")
    val_dataset = MMFiPoseDataset(root, data_config, "val")
    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 32))
    num_workers = int(data_config.get("num_workers", 4))
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
