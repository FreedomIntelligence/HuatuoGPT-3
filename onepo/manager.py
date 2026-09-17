"""OnePO rollout manager for current verl.

injects cached or API teacher outputs,
scores them with verl's reward workers, applies Teacher Retirement, and keeps a
fixed rollout group size by uniformly replacing one on-policy trajectory.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

import aiohttp
import numpy as np
import torch
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

from onepo.retirement import RetirementController, retained_teacher_indices


class TeacherGenerator:
    """OpenAI-compatible async teacher client with bounded concurrency."""

    def __init__(self, config: Any):
        self.online_enabled = str(config.get("online_enabled", False)).lower() == "true"
        self.endpoint = str(config.get("api_url", os.environ.get("ONEPO_TEACHER_API_URL", ""))).rstrip("/")
        if self.endpoint and not self.endpoint.endswith("chat/completions"):
            self.endpoint += "/chat/completions"
        self.api_key = os.environ.get("ONEPO_TEACHER_API_KEY", "")
        self.model = str(config.get("model", os.environ.get("ONEPO_TEACHER_MODEL", "gpt-5.4")))
        self.max_tokens = int(config.get("max_tokens", 8192))
        self.timeout = float(config.get("timeout", 180))
        self.max_retries = int(config.get("max_retries", 3))
        self.semaphore = asyncio.Semaphore(int(config.get("max_concurrency", 64)))

    async def generate(self, messages: list[dict[str, str]]) -> str | None:
        if not self.online_enabled or not self.endpoint or not self.api_key:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    async with (
                        aiohttp.ClientSession(timeout=timeout) as session,
                        session.post(self.endpoint, headers=headers, json=payload) as response,
                    ):
                        response.raise_for_status()
                        message = (await response.json())["choices"][0]["message"]
                        content = message.get("content") or ""
                        reasoning = message.get("reasoning_content")
                        return f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, TypeError, json.JSONDecodeError):
                    if attempt + 1 == self.max_retries:
                        return None
                    await asyncio.sleep(min(2**attempt, 8) + random.random())
        return None


class OnePOAgentLoopManager(AgentLoopManager):
    """Batch-level OnePO extension for verl's single-turn rollout manager."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        onepo = config.get("onepo", {})
        retirement = onepo.get("retirement", {})
        self.controller = RetirementController(
            max_teacher_steps=int(onepo.get("teacher_max_steps", -1)),
            plateau_window=int(retirement.get("plateau_window", -1)),
            plateau_tolerance=float(retirement.get("plateau_tolerance", 0.01)),
            plateau_min_retirement_rate=float(retirement.get("plateau_min_retirement_rate", 0.9)),
        )
        self.teacher_enabled_by_config = bool(onepo.get("teacher_enabled", True))
        self.teacher = TeacherGenerator(onepo.get("teacher", {}))
        tokenizer_path = config.actor_rollout_ref.model.get("tokenizer_path") or config.actor_rollout_ref.model.path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.seed = int(onepo.get("seed", 42))

    @staticmethod
    def _reason_code(reason: str | None) -> int:
        if not reason:
            return 0
        if reason.startswith("max_teacher_steps"):
            return 1
        if reason.startswith("retirement plateau"):
            return 2
        if reason == "validation":
            return 3
        if reason == "disabled_by_config":
            return 4
        return 99

    def _publish_metrics(self, output: DataProto, step: int, *, enabled: bool, reason: str | None,
                         candidates: int = 0, replaced: int = 0, retired: int | None = None) -> None:
        """Attach numeric metrics to the output; the driver logs this namespace verbatim."""
        if retired is None:
            retired = max(0, candidates - replaced)
        teacher_tokens = int(output.batch["onepo_teacher_mask"].sum().item())
        response_tokens = int(output.batch["response_mask"].sum().item())
        ratio = replaced / candidates if candidates else 0.0
        retirement_rate = retired / candidates if candidates else (1.0 if not enabled else 0.0)
        metrics = {
            "OnePO/teacher_enabled": float(enabled),
            "OnePO/teacher_disabled_reason_code": float(self._reason_code(reason)),
            "OnePO/teacher_candidates": float(candidates),
            "OnePO/teacher_replaced": float(replaced),
            "OnePO/replaced_count": float(replaced),
            "OnePO/retired_count": float(retired),
            "OnePO/skipped_count": float(max(0, candidates - replaced)),
            "OnePO/replacement_ratio": float(ratio),
            "OnePO/retirement_rate": float(retirement_rate),
            "OnePO/teacher_tokens": float(teacher_tokens),
            "OnePO/teacher_token_fraction": teacher_tokens / response_tokens if response_tokens else 0.0,
            "OnePO/response_tokens": float(response_tokens),
        }
        output.meta_info["onepo_metrics"] = metrics
        print(json.dumps({"onepo_step": step, **metrics, "teacher_disabled_reason": reason}, ensure_ascii=True), flush=True)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        output = await super().generate_sequences(prompts)
        response_mask = output.batch["response_mask"]
        teacher_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        output.batch["onepo_teacher_mask"] = teacher_mask
        output.non_tensor_batch["response_source"] = np.array(["model"] * len(output), dtype=object)

        step = int(prompts.meta_info.get("global_steps", -1))
        validate = bool(prompts.meta_info.get("validate", False))
        if validate:
            self._publish_metrics(output, step, enabled=False, reason="validation")
            return output
        if not self.teacher_enabled_by_config:
            self._publish_metrics(output, step, enabled=False, reason="disabled_by_config")
            return output
        if not self.controller.teacher_enabled(step):
            self._publish_metrics(output, step, enabled=False, reason=self.controller.disabled_reason)
            return output
        if self.reward_loop_worker_handles is None:
            raise RuntimeError("OnePO requires verl reward-loop workers to score teacher candidates")

        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if len(output) % rollout_n:
            raise ValueError(f"rollout batch {len(output)} is not divisible by rollout.n={rollout_n}")
        group_starts = list(range(0, len(output), rollout_n))
        teacher_texts = await self._teacher_texts(prompts, group_starts)
        candidates: list[tuple[int, DataProto]] = []
        for group_start, text in zip(group_starts, teacher_texts, strict=True):
            if text:
                candidates.append((group_start, self._build_teacher_candidate(output, prompts, group_start, text)))

        score_tasks = []
        for i, (_, candidate) in enumerate(candidates):
            handle = self.reward_loop_worker_handles[i % len(self.reward_loop_worker_handles)]
            score_tasks.append(handle.compute_score.remote(candidate))
        score_results = await asyncio.gather(*score_tasks) if score_tasks else []

        rng = np.random.default_rng(self.seed + max(step, 0))
        replaced = 0
        for (group_start, candidate), result in zip(candidates, score_results, strict=True):
            group_slice = slice(group_start, group_start + rollout_n)
            model_rewards = output.batch["rm_scores"][group_slice].sum(dim=-1)
            # Compare at the same precision used to store training rewards.
            # Keep strict >: no rounding tolerance that hides real improvements.
            teacher_reward = torch.as_tensor(result["reward_score"], dtype=model_rewards.dtype).item()
            if not retained_teacher_indices([teacher_reward], model_rewards.cpu().tolist()):
                continue
            replace_idx = int(rng.integers(group_start, group_start + rollout_n))
            self._replace_row(output, candidate, replace_idx, teacher_reward, result.get("reward_extra_info", {}))
            replaced += 1

        total = len(candidates)
        retired = total - replaced
        self.controller.observe(retired=retired, total=total)
        self._publish_metrics(
            output, step, enabled=True, reason=self.controller.disabled_reason,
            candidates=total, replaced=replaced, retired=retired,
        )
        return output

    async def _teacher_texts(self, prompts: DataProto, group_starts: list[int]) -> list[str | None]:
        cached = prompts.non_tensor_batch.get("teacher_response")
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        tasks = []
        positions = []
        result: list[str | None] = [None] * len(group_starts)
        for out_idx, group_start in enumerate(group_starts):
            value = cached[group_start] if cached is not None else None
            if isinstance(value, np.ndarray):
                value = value.item()
            if isinstance(value, str) and value.strip():
                result[out_idx] = value
            elif raw_prompts is not None and self.teacher.online_enabled:
                tasks.append(self.teacher.generate(list(raw_prompts[group_start])))
                positions.append(out_idx)
        generated = await asyncio.gather(*tasks) if tasks else []
        for out_idx, text in zip(positions, generated, strict=True):
            result[out_idx] = text
        return result

    def _build_teacher_candidate(
        self, output: DataProto, prompts: DataProto, source_idx: int, response_text: str
    ) -> DataProto:
        batch = output.batch[source_idx : source_idx + 1].clone()
        for key in ("rm_scores", "rollout_log_probs", "routed_experts", "onepo_teacher_mask"):
            if key in batch:
                del batch[key]

        response_width = output.batch["responses"].shape[-1]
        prompt_width = output.batch["prompts"].shape[-1]
        prompt_ids = output.batch["prompts"][source_idx].clone()
        prompt_attention = output.batch["attention_mask"][source_idx, :prompt_width].clone()
        think_prefilled = self.tokenizer.decode(
            prompt_ids[prompt_attention.bool()], skip_special_tokens=False
        ).rstrip().endswith("<think>")
        if think_prefilled and response_text.lstrip().startswith("<think>"):
            response_text = response_text.lstrip()[len("<think>"):].lstrip("\r\n")
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)[:response_width]
        if self.tokenizer.eos_token_id is not None and len(response_ids) < response_width:
            response_ids.append(self.tokenizer.eos_token_id)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        padded = torch.full((response_width,), pad_id, dtype=torch.long)
        mask = torch.zeros((response_width,), dtype=torch.long)
        padded[: len(response_ids)] = torch.tensor(response_ids, dtype=torch.long)
        mask[: len(response_ids)] = 1

        attention = torch.cat([prompt_attention, mask])
        batch["responses"][0] = padded
        batch["response_mask"][0] = mask
        batch["input_ids"][0] = torch.cat([prompt_ids, padded])
        batch["attention_mask"][0] = attention
        positions = (attention.cumsum(dim=-1) - 1).clamp_min(0)
        if batch["position_ids"].ndim == 2:
            batch["position_ids"][0] = positions
        elif batch["position_ids"].ndim == 3:
            # Text-only Qwen VL inputs have identical positions on every RoPE axis.
            # Preserve prompt positions and extend each axis for the new response.
            prompt_positions = batch["position_ids"][0, :, :prompt_width]
            valid_positions = prompt_positions[:, prompt_attention.bool()]
            if valid_positions.numel() == 0 or not torch.equal(
                valid_positions, valid_positions[:1].expand_as(valid_positions)
            ):
                raise NotImplementedError("OnePO Teacher replacement supports text-only RoPE inputs")
            response_positions = valid_positions[:, -1:] + torch.arange(
                1, response_width + 1, device=prompt_positions.device
            )
            batch["position_ids"][0, :, prompt_width:] = response_positions.masked_fill(~mask.bool(), 0)
        else:
            raise NotImplementedError("Unsupported OnePO position_ids shape")

        non_tensor = {
            key: np.array([values[source_idx]], dtype=object) for key, values in prompts.non_tensor_batch.items()
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})

    @staticmethod
    def _replace_row(
        output: DataProto,
        candidate: DataProto,
        target_idx: int,
        teacher_reward: float,
        reward_extra_info: dict[str, Any],
    ) -> None:
        for key in ("responses", "response_mask", "input_ids", "attention_mask", "position_ids"):
            output.batch[key][target_idx].copy_(candidate.batch[key][0])
        if "rollout_log_probs" in output.batch:
            del output.batch["rollout_log_probs"]
        if "routed_experts" in output.batch:
            del output.batch["routed_experts"]
        output.batch["rm_scores"][target_idx].zero_()
        valid = int(output.batch["response_mask"][target_idx].sum().item())
        output.batch["rm_scores"][target_idx, max(valid - 1, 0)] = teacher_reward
        output.batch["onepo_teacher_mask"][target_idx].copy_(output.batch["response_mask"][target_idx].bool())
        output.non_tensor_batch["response_source"][target_idx] = "teacher"
        for key, value in reward_extra_info.items():
            if key in output.non_tensor_batch:
                if output.non_tensor_batch[key].dtype.kind in "US":
                    output.non_tensor_batch[key] = output.non_tensor_batch[key].astype(object)
                output.non_tensor_batch[key][target_idx] = value
