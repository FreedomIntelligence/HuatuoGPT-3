"""OnePO training entry point for current verl.

composes verl's native configuration and
runs the custom task runner while preserving upstream validation and lifecycle.
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from onepo.trainer import OnePOTaskRunner


def worker_runtime_env(config: DictConfig) -> dict:
    """Explicitly propagate OnePO settings even when attaching to an existing Ray cluster.

    Keep credentials out of Hydra config, which verl prints to the training log.
    Nested workers inherit this task runner's runtime environment.
    """
    names = (
        "ONEPO_PROBABILITY_FLOOR", "REQUIRE_THINKING",
        "ANSWER_EXPECTED_TOKENS", "ANSWER_BUFFER_TOKENS", "ANSWER_PENALTY_FACTOR",
        "ONEPO_TOKENIZER_PATH",
        "REWARD_MODEL_URL", "REWARD_MODEL_NAME", "REWARD_MODEL_TIMEOUT",
        "REWARD_MODEL_MAX_TOKENS", "REWARD_MODEL_MAX_RETRIES",
        "ONEPO_TEACHER_API_URL", "ONEPO_TEACHER_API_KEY", "ONEPO_TEACHER_MODEL",
        "PYTHONPATH", "SWANLAB_LOG_DIR", "TMPDIR", "RAY_TMPDIR",
    )
    env = {name: os.environ[name] for name in names if name in os.environ}
    env["ONEPO_PROBABILITY_FLOOR"] = str(float(config.onepo.probability_floor))
    model_config = OmegaConf.select(config, "actor_rollout_ref.model")
    if model_config is not None:
        env["ONEPO_TOKENIZER_PATH"] = str(model_config.get("tokenizer_path") or model_config.path)
    return {"env_vars": env}


class _RuntimeEnvRunner:
    """Preserve worker env when verl adds runtime options such as Nsight."""

    def __init__(self, runtime_env):
        self.runtime_env = runtime_env

    def remote(self):
        return self.options().remote()

    def options(self, **options):
        extra = options.pop("runtime_env", {})
        runtime_env = {**self.runtime_env, **extra}
        runtime_env["env_vars"] = {**self.runtime_env["env_vars"], **extra.get("env_vars", {})}
        return OnePOTaskRunner.options(runtime_env=runtime_env, **options)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    floor = float(config.onepo.probability_floor)
    os.environ["ONEPO_PROBABILITY_FLOOR"] = str(floor)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    runner = _RuntimeEnvRunner(worker_runtime_env(config))
    run_ppo(config, task_runner_class=runner)


if __name__ == "__main__":
    main()


# Run: bash OnePO.sh
