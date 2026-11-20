from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .custom_manifest import ManifestPoseDataset
from .mmfi import MMFiPoseDataset
from .person_in_wifi_3d import PersonInWiFi3DPoseDataset


def build_dataset(config: dict[str, Any], split: str, data_root: str | Path | None = None) -> Dataset:
    data_config = config["data"]
    root = data_root if data_root is not None else data_config["root"]
    dataset_name = str(data_config.get("dataset", "mmfi")).lower()
    if dataset_name == "mmfi":
        return MMFiPoseDataset(root, data_config, split)
    if dataset_name in {"person_in_wifi_3d", "person-in-wifi-3d", "piwifi3d"}:
        return PersonInWiFi3DPoseDataset(root, data_config, split)
    if dataset_name in {"custom", "custom_manifest", "real_world"}:
        return ManifestPoseDataset(root, data_config, split)
    raise ValueError(
        "data.dataset must be one of: mmfi, person_in_wifi_3d, custom_manifest."
    )


def build_dataloaders(
    config: dict[str, Any],
    data_root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset(config, "train", data_root)
    val_dataset = build_dataset(config, "val", data_root)
    training = config.get("training", {})
    data_config = config["data"]
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
