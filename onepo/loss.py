"""Paper-aligned OnePO policy loss.

registers Adaptive Objective Evolution
with verl while keeping the standard on-policy PPO/DAPO objective unchanged.
"""

from __future__ import annotations

import math
import os
from contextvars import ContextVar
from typing import Any

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import get_policy_loss_fn, register_policy_loss
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


# verl's policy-loss registry does not accept additional batch fields. Bridge
# the mask within one synchronous loss call, without mutating verl or config.
_teacher_mask: ContextVar[torch.Tensor | None] = ContextVar("onepo_teacher_mask", default=None)


def onepo_ppo_loss(config, model_output, data, dp_group=None):
    if "onepo_teacher_mask" not in data:
        raise RuntimeError("OnePO loss requires an explicit onepo_teacher_mask from OnePOAgentLoopManager")
    mask = data.select("onepo_teacher_mask").to_padded_tensor()["onepo_teacher_mask"]
    token = _teacher_mask.set(mask)
    try:
        return ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)
    finally:
        _teacher_mask.reset(token)


def probability_floor() -> float:
    value = float(os.environ.get("ONEPO_PROBABILITY_FLOOR", "0.1"))
    if not 0 < value < 1:
        raise ValueError("ONEPO_PROBABILITY_FLOOR must be in (0, 1)")
    return value


@register_policy_loss("onepo")
def compute_onepo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: ActorConfig | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    teacher_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute standard clipped PPO plus OnePO's teacher-token rescaling.

    Keep batch old_log_probs raw; floor only the local teacher denominator.
    A positive detached multiplier can be moved inside PPO onto advantages
    without changing clipping branches. Delegate clipping, dual clipping,
    the stability clamp and aggregation to verl's native vanilla objective.
    """
    if config is None:
        raise ValueError("OnePO policy loss requires an ActorConfig")

    floor = probability_floor()
    if teacher_mask is None:
        teacher_mask = _teacher_mask.get()
    if teacher_mask is None:
        raise RuntimeError("OnePO policy loss must be called through onepo_ppo_loss or with an explicit mask")
    if teacher_mask.shape != response_mask.shape:
        raise ValueError("OnePO teacher mask must have the same shape as response_mask")
    teacher_mask = teacher_mask.to(device=log_prob.device, dtype=torch.bool) & response_mask.bool()
    log_floor = math.log(floor)
    low_support_mask = teacher_mask & (old_log_prob < log_floor)
    effective_old_log_prob = torch.where(teacher_mask, old_log_prob.clamp_min(log_floor), old_log_prob)
    positive_floor_mask = low_support_mask & (advantages > 0)
    negative_floor_mask = low_support_mask & (advantages <= 0)
    rescale = floor / torch.exp(log_prob.detach()).clamp_min(torch.finfo(log_prob.dtype).tiny)
    effective_advantages = advantages * torch.where(positive_floor_mask, rescale, 1.0)

    pg_loss, metrics = get_policy_loss_fn("vanilla")(
        old_log_prob=effective_old_log_prob,
        log_prob=log_prob,
        advantages=effective_advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    response_tokens = response_mask.bool().sum().clamp_min(1)
    low_support_tokens = low_support_mask.sum()
    positive_tokens = positive_floor_mask.sum()
    negative_tokens = negative_floor_mask.sum()
    metrics.update({
        "OnePO/rescaled_token_fraction": (positive_tokens.float() / response_tokens).detach().item(),
        "OnePO/rescaled_low_support_fraction": (
            positive_tokens.float() / low_support_tokens.clamp_min(1)
        ).detach().item(),
        "OnePO/low_support_tokens": float(low_support_tokens.detach().item()),
        "OnePO/positive_advantage_teacher_tokens": float(positive_tokens.detach().item()),
        "OnePO/negative_advantage_teacher_tokens": float(negative_tokens.detach().item()),
        "OnePO/floored_teacher_tokens": float(low_support_tokens.detach().item()),
        "OnePO/rescale_mean": verl_F.masked_mean(rescale, positive_floor_mask).detach().item(),
    })
    return pg_loss, metrics
