"""Build OnePO's loss adapter for verl's public WorkerGroup.set_loss_fn API."""

from functools import partial

from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass

from onepo.loss import onepo_ppo_loss


def build_onepo_loss(config):
    """Construct the loss configuration explicitly, without worker internals."""
    if is_distillation_enabled(config.get("distillation")):
        raise NotImplementedError("OnePO's loss adapter does not support verl distillation")
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    if actor_config.policy_loss.loss_mode != "onepo":
        raise ValueError("OnePO requires actor_rollout_ref.actor.policy_loss.loss_mode=onepo")
    return partial(onepo_ppo_loss, config=actor_config)
