from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import total_objective
from .metrics import pose_metrics
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    training_config = config.get("training", {})
    sums = {"total": 0.0, "pose": 0.0, "action": 0.0, "domain": 0.0}
    sample_count = 0

    for batch in loader:
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        static = batch["static"].to(device, non_blocking=True)
        poses = batch["pose"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        environments = batch["environment"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(dynamic, static, environments)
        loss, components = total_objective(
            outputs,
            poses,
            actions,
            environments,
            action_weight=float(training_config.get("action_loss_weight", 0.5)),
            domain_weight=float(training_config.get("domain_loss_weight", 0.2)),
            temperature=float(training_config.get("contrastive_temperature", 0.07)),
        )
        loss.backward()
        optimizer.step()

        batch_size = dynamic.shape[0]
        sample_count += batch_size
        for name in sums:
            sums[name] += float(components[name].item()) * batch_size

    if sample_count == 0:
        raise RuntimeError("The training loader produced no batches. Check the split and batch size.")
    return {name: value / sample_count for name, value in sums.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    unknown_environment: bool = False,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    action_correct = 0
    sample_count = 0

    for batch in loader:
        dynamic = batch["dynamic"].to(device, non_blocking=True)
        static = batch["static"].to(device, non_blocking=True)
        poses = batch["pose"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        environments = batch["environment"].to(device, non_blocking=True)
        if unknown_environment:
            environments = torch.full_like(environments, -1)

        outputs = model(dynamic, static, environments)
        predictions.append(outputs["pose"].cpu().numpy())
        targets.append(poses.cpu().numpy())
        action_correct += int(outputs["action_logits"].argmax(dim=-1).eq(actions).sum().item())
        sample_count += dynamic.shape[0]

    if sample_count == 0:
        raise RuntimeError("The evaluation loader produced no samples.")
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    metric_config = config.get("metrics", {})
    result = pose_metrics(
        prediction_array,
        target_array,
        right_shoulder_index=int(metric_config.get("right_shoulder_index", 5)),
        left_hip_index=int(metric_config.get("left_hip_index", 12)),
    )
    result["action_accuracy"] = action_correct / sample_count
    result["num_samples"] = float(sample_count)
    return result


def run_training(
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Path:
    training_config = config.get("training", {})
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        betas=(0.9, 0.999),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_config.get("lr_step_size", 30)),
        gamma=float(training_config.get("lr_gamma", 0.5)),
    )

    checkpoint_dir = Path(training_config.get("checkpoint_dir", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    best_mpjpe = float("inf")
    epochs = int(training_config.get("epochs", 100))
    unknown_environment = str(config["data"].get("split", "")).lower() == "cross_environment"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            config,
            unknown_environment=unknown_environment,
        )
        scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['total']:.5f} "
            f"(pose {train_metrics['pose']:.5f}, action {train_metrics['action']:.5f}, "
            f"domain {train_metrics['domain']:.5f}) | "
            f"val MPJPE {val_metrics['mpjpe_mm']:.2f} mm | "
            f"PA-MPJPE {val_metrics['pa_mpjpe_mm']:.2f} mm | "
            f"PCK@20 {100.0 * val_metrics['pck_20']:.2f}% | "
            f"PCK@50 {100.0 * val_metrics['pck_50']:.2f}% | lr {learning_rate:.2e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mpjpe_mm": min(best_mpjpe, val_metrics["mpjpe_mm"]),
            "config": config,
        }
        if training_config.get("save_last", True):
            torch.save(checkpoint, last_path)
        if val_metrics["mpjpe_mm"] < best_mpjpe:
            best_mpjpe = val_metrics["mpjpe_mm"]
            checkpoint["best_mpjpe_mm"] = best_mpjpe
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path} ({best_mpjpe:.2f} mm)")

    return best_path
