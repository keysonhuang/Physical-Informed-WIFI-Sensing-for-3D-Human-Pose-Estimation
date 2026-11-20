from __future__ import annotations

import torch
from torch.nn import functional as F


def pose_regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean joint displacement."""
    return (prediction - target).square().sum(dim=-1).mean()


def supervised_cross_environment_loss(
    features: torch.Tensor,
    actions: torch.Tensor,
    environments: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrast same-action features from different environments."""
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim].")
    batch_size = features.shape[0]
    if batch_size < 2:
        return features.sum() * 0.0

    normalized = F.normalize(features, dim=-1)
    similarities = torch.matmul(normalized, normalized.transpose(0, 1)) / temperature
    non_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive = (
        actions.view(-1, 1).eq(actions.view(1, -1))
        & environments.view(-1, 1).ne(environments.view(1, -1))
        & non_self
    )
    valid = positive.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0

    logits = similarities.masked_fill(~non_self, -1e4)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (log_probabilities * positive).sum(dim=1) / positive_count
    return -mean_positive_log_probability[valid].mean()


def total_objective(
    outputs: dict[str, torch.Tensor],
    pose_target: torch.Tensor,
    action_target: torch.Tensor,
    environment_target: torch.Tensor,
    action_weight: float = 0.5,
    domain_weight: float = 0.2,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pose_loss = pose_regression_loss(outputs["pose"], pose_target)
    action_loss = F.cross_entropy(outputs["action_logits"], action_target)
    domain_loss = supervised_cross_environment_loss(
        outputs["dynamic_feature"],
        action_target,
        environment_target,
        temperature=temperature,
    )
    total = pose_loss + action_weight * action_loss + domain_weight * domain_loss
    return total, {
        "pose": pose_loss.detach(),
        "action": action_loss.detach(),
        "domain": domain_loss.detach(),
        "total": total.detach(),
    }
