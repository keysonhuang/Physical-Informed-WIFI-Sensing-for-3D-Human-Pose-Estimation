from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wifi_pose.config import load_config
from wifi_pose.data.mmfi import MMFiPoseDataset
from wifi_pose.engine import evaluate_model, resolve_device
from wifi_pose.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WiFi-based 3D pose estimator.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default=None, help="Optional YAML configuration override.")
    parser.add_argument("--data-root", default=None, help="Override the dataset root directory.")
    parser.add_argument("--device", default="auto", help="Device such as auto, cpu, or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional evaluation batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else checkpoint["config"]
    if args.data_root:
        config["data"]["root"] = args.data_root

    dataset = MMFiPoseDataset(config["data"]["root"], config["data"], split="val")
    batch_size = args.batch_size or int(config.get("training", {}).get("batch_size", 32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    unknown_environment = str(config["data"].get("split", "")).lower() == "cross_environment"
    metrics = evaluate_model(model, loader, device, config, unknown_environment=unknown_environment)

    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Samples: {int(metrics['num_samples']):,}")
    print(f"MPJPE: {metrics['mpjpe_mm']:.2f} mm")
    print(f"PA-MPJPE: {metrics['pa_mpjpe_mm']:.2f} mm")
    print(f"PCK@20: {100.0 * metrics['pck_20']:.2f}%")
    print(f"PCK@50: {100.0 * metrics['pck_50']:.2f}%")
    print(f"Action accuracy: {100.0 * metrics['action_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
