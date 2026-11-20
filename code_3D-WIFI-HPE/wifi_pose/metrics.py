from __future__ import annotations

import numpy as np
import torch


def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def mpjpe(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    prediction, target = _as_numpy(prediction), _as_numpy(target)
    return float(np.linalg.norm(prediction - target, axis=-1).mean())


def pa_mpjpe(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    prediction = _as_numpy(prediction).astype(np.float64, copy=False)
    target = _as_numpy(target).astype(np.float64, copy=False)
    if prediction.ndim == 2:
        prediction, target = prediction[None, ...], target[None, ...]
    errors = []
    for predicted_pose, target_pose in zip(prediction, target):
        predicted_mean = predicted_pose.mean(axis=0, keepdims=True)
        target_mean = target_pose.mean(axis=0, keepdims=True)
        predicted_centered, target_centered = predicted_pose - predicted_mean, target_pose - target_mean
        covariance = predicted_centered.T @ target_centered
        u, singular_values, vt = np.linalg.svd(covariance)
        correction = np.eye(3, dtype=np.float64)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        scale = np.sum(singular_values * np.diag(correction)) / max(np.sum(predicted_centered**2), 1e-12)
        aligned = scale * (predicted_centered @ rotation) + target_mean
        errors.append(np.linalg.norm(aligned - target_pose, axis=-1).mean())
    return float(np.mean(errors))


def pck(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    threshold_percent: float,
    right_shoulder_index: int = 5,
    left_hip_index: int = 12,
) -> float:
    prediction, target = _as_numpy(prediction), _as_numpy(target)
    if prediction.ndim == 2:
        prediction, target = prediction[None, ...], target[None, ...]
    shoulder, hip = target[:, right_shoulder_index], target[:, left_hip_index]
    torso_length = np.linalg.norm(shoulder - hip, axis=-1)
    valid = torso_length > 1e-8
    if not np.any(valid):
        return 0.0
    distances = np.linalg.norm(prediction - target, axis=-1)
    return float((distances[valid] / torso_length[valid, None] <= threshold_percent / 100.0).mean())


def pose_metrics(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    right_shoulder_index: int | None = 5,
    left_hip_index: int | None = 12,
) -> dict[str, float]:
    result = {"mpjpe_mm": mpjpe(prediction, target), "pa_mpjpe_mm": pa_mpjpe(prediction, target)}
    if right_shoulder_index is not None and left_hip_index is not None:
        result["pck_20"] = pck(prediction, target, 20, right_shoulder_index, left_hip_index)
        result["pck_50"] = pck(prediction, target, 50, right_shoulder_index, left_hip_index)
    return result
