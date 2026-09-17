"""Minimal OnePO integration with verl's current PPO trainer.

Wire the native workers, DAPO trainer and public loss adapter. OnePO's
probability floor is local to the loss; trainer old_log_probs stay unmodified.
"""

from __future__ import annotations

from pprint import pprint

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config import HFModelConfig
from verl.workers.engine_workers import ActorRolloutRefWorker

from onepo.dapo_trainer import RayDAPOTrainer
from onepo.evaluation import should_validate
from onepo.worker import build_onepo_loss


# Preserve the import name without overriding any upstream trainer hooks.
OnePOTrainer = RayDAPOTrainer


@ray.remote
class OnePOTaskRunner(BaseTaskRunner):
    """Build current verl workers while selecting OnePO's narrow extensions."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return ActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        should_validate(0, config.trainer.test_freq, config.trainer.get("test_steps", []))
        loss_fn = build_onepo_loss(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        from verl.utils.dataset.rl_dataset import collate_fn

        trainer = OnePOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer.init_workers()
        trainer.actor_rollout_wg.set_loss_fn(loss_fn)
        trainer.fit()


# Run: python -m onepo.train --help
