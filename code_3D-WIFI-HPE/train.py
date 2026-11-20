from __future__ import annotations

import argparse
from pathlib import Path

from wifi_pose.config import load_config
from wifi_pose.data import build_dataloaders
from wifi_pose.engine import resolve_device, run_training, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WiFi-based 3D pose estimator.")
    parser.add_argument("--config", default="configs/mmfi.yaml", help="Path to the YAML configuration.")
    parser.add_argument("--data-root", default=None, help="Override the dataset root directory.")
    parser.add_argument("--device", default="auto", help="Device such as auto, cpu, or cuda:0.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override the checkpoint output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.checkpoint_dir:
        config.setdefault("training", {})["checkpoint_dir"] = args.checkpoint_dir
    train_loader, val_loader = build_dataloaders(config)
    print(f"Training frames: {len(train_loader.dataset):,}")
    print(f"Validation frames: {len(val_loader.dataset):,}")
    device = resolve_device(args.device)
    print(f"Device: {device}")
    best_checkpoint = run_training(config, train_loader, val_loader, device)
    print(f"Training complete. Best checkpoint: {Path(best_checkpoint).resolve()}")


if __name__ == "__main__":
    main()
