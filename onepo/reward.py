"""Mixed verifiable and HealthBench-style rubric reward for OnePO."""

from __future__ import annotations

import json
import math
import os
import re
import time
from functools import lru_cache
from typing import Any

import requests
from transformers import AutoTokenizer

ANSWER_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", flags=re.IGNORECASE)

SCORE_TEMPLATE = """Score the assistant's response against each rubric item.

## Conversation
<Conversation>
{conversation}
</Conversation>

## Rubrics
<Rubric_items>
{rubrics}
</Rubric_items>

## Output
Return only a JSON list of booleans, one for each rubric in order.
- true: the criterion is met
- false: the criterion is not met
- For a negative criterion (bad behavior with negative points), return true only when the response shows that bad behavior.
""".strip()


def extract_choice(text: str) -> str | None:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    fallback = re.search(r"(?:answer is|answer:|答案(?:是|为|：))\s*\(?([A-Za-z])\)?", text, flags=re.IGNORECASE)
    return fallback.group(1).upper() if fallback else None


def remove_thinking_part(text: str) -> str:
    """Return only the user-visible answer passed to the rubric grader."""
    if "## Final Response\n\n" in text:
        text = text.rsplit("## Final Response\n\n", 1)[-1]
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


@lru_cache(maxsize=1)
def _answer_tokenizer(model_path: str):
    """Reuse the student tokenizer within each reward worker."""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def answer_length_penalty(solution: str) -> tuple[float, int]:
    """Capped final-answer length penalty; independent of total rollout length."""
    expected = int(os.environ.get("ANSWER_EXPECTED_TOKENS", "200"))
    buffer = int(os.environ.get("ANSWER_BUFFER_TOKENS", "4000"))
    factor = float(os.environ.get("ANSWER_PENALTY_FACTOR", "0.5"))
    if expected < 0 or buffer <= 0 or not math.isfinite(factor) or factor < 0:
        raise ValueError("Answer penalty requires expected >= 0, buffer > 0 and finite factor >= 0")
    if factor == 0:
        return 0.0, 0
    # Count the full final answer, not the character-truncated grading prompt.
    answer = remove_thinking_part(solution)
    if not answer:
        return 0.0, 0
    model_path = os.environ.get("ONEPO_TOKENIZER_PATH", "")
    if not model_path:
        raise ValueError("ONEPO_TOKENIZER_PATH must point to the student tokenizer")
    tokens = len(_answer_tokenizer(model_path).encode(answer, add_special_tokens=False))
    exceed = min(max(tokens - expected, 0), buffer)
    return -factor * exceed / buffer, tokens


def _parse_boolean_list(text: str, expected: int) -> list[bool] | None:
    candidates = [text]
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            values = json.loads(candidate.lower())
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(values, list) and len(values) == expected and all(isinstance(v, bool) for v in values):
            return values
    return None


def _conversation_text(prompt: Any, response_text: str) -> str:
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    conversation = list(messages) + [{"role": "assistant", "content": response_text}]
    return "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        if isinstance(message, dict)
        else f"user: {message}"
        for message in conversation
    )


def _rubrics_text(rubrics: list[Any]) -> str:
    lines = []
    for index, item in enumerate(rubrics, start=1):
        if isinstance(item, dict):
            points = float(item.get("points", 0))
            criterion = item.get("criterion", "")
        else:
            points = 0.0
            criterion = str(item)
        lines.append(f"{index}. ({points:+g}pts) {criterion}")
    return "\n".join(lines)


def _request_grading(prompt: str, temperature: float) -> str:
    server = os.environ.get("REWARD_MODEL_URL", "http://localhost:30000/v1/chat/completions")
    model = os.environ.get("REWARD_MODEL_NAME", "default")
    timeout = float(os.environ.get("REWARD_MODEL_TIMEOUT", "120"))
    max_tokens = int(os.environ.get("REWARD_MODEL_MAX_TOKENS", "256"))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # requests sends JSON directly; SDK extra_body nesting is not a wire field.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Retry the entire request + parse together in _rubric_score, so mixed
    # network/format failures share one bounded retry budget.
    response = requests.post(server, json=payload, timeout=timeout)
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"] or "")


def _rubric_score(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    rubrics = ground_truth.get("rubrics") or []
    if not rubrics:
        return {"score": 0.0, "acc": 0.0, "success": False, "error": "missing_rubrics", "predicted_answer": ""}

    response_text = remove_thinking_part(solution)[:10000]
    prompt = SCORE_TEMPLATE.format(
        conversation=_conversation_text(ground_truth.get("prompt", []), response_text),
        rubrics=_rubrics_text(rubrics),
    )

    max_retries = max(0, int(os.environ.get("REWARD_MODEL_MAX_RETRIES", "2")))
    temperatures = (0.0, 0.2, 0.4)
    content = ""
    decisions = None
    error = "invalid_grader_output"
    for attempt in range(max_retries + 1):
        try:
            content = _request_grading(prompt, temperatures[min(attempt, len(temperatures) - 1)])
            decisions = _parse_boolean_list(content, len(rubrics))
            if decisions is not None:
                break
            error = "invalid_grader_output"
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            error = type(exc).__name__
        if attempt < max_retries:
            time.sleep(min(2**attempt, 4))

    if decisions is None:
        server = os.environ.get("REWARD_MODEL_URL", "http://localhost:30000/v1/chat/completions")
        print(
            "!!!!! WARNING: ONEPO RUBRIC GRADING FAILED; USING ZERO REWARD !!!!! "
            f"server={server} attempts={max_retries + 1} error={error} "
            f"rubrics={len(rubrics)} response={content[:300]!r}",
            flush=True,
        )
        return {
            "score": 0.0,
            "acc": 0.0,
            "success": False,
            "error": error,
            "predicted_answer": "",
        }

    earned = sum(float(item.get("points", 0)) for item, passed in zip(rubrics, decisions) if passed)
    positive_total = sum(max(float(item.get("points", 0)), 0.0) for item in rubrics)
    score = max(0.0, min(1.0, earned / positive_total)) if positive_total else 0.0
    return {
        "score": score,
        "acc": score,
        "success": True,
        "rubric_hits": sum(decisions),
        "predicted_answer": "",
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Route a sample to its verifiable or rubric-based reward."""

    def _finalize(result: dict[str, Any]) -> dict[str, Any]:
        penalty, answer_tokens = 0.0, 0
        # Preserve existing format-error and exhausted-retry zero-score behavior.
        if result.get("success", False) and not result.get("error"):
            penalty, answer_tokens = answer_length_penalty(solution_str)
        # verl concatenates reward metadata and requires identical key order.
        return {
            "score": result.get("score", 0.0) + penalty,
            "acc": result.get("acc", 0.0),
            "success": result.get("success", False),
            "error": result.get("error", ""),
            "predicted_answer": result.get("predicted_answer", ""),
            "rubric_hits": result.get("rubric_hits", 0),
            "answer_length_penalty": penalty,
            "answer_tokens": answer_tokens,
        }

    require_thinking = os.environ.get("REQUIRE_THINKING", "false").lower() == "true"
    think_prefilled = (extra_info or {}).get("prompt_ends_with_think", False) is True
    if require_thinking and not ((think_prefilled or "<think>" in solution_str) and "</think>" in solution_str):
        return _finalize(
            {
                "score": 0.0,
                "acc": 0.0,
                "success": True,
                "error": "missing_thinking_tags",
                "predicted_answer": "",
            }
        )
    if ground_truth.get("type") == "openend":
        try:
            result = _rubric_score(solution_str, ground_truth)
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            server = os.environ.get("REWARD_MODEL_URL", "http://localhost:30000/v1/chat/completions")
            print(
                "!!!!! WARNING: ONEPO RUBRIC REQUEST FAILED !!!!! "
                f"server={server} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            return _finalize(
                {
                    "score": 0.0,
                    "acc": 0.0,
                    "success": False,
                    "error": type(exc).__name__,
                    "predicted_answer": "",
                }
            )

        # Configuration/tokenizer failures must not be reported as grader failures.
        return _finalize(result)

    expected = str(ground_truth.get("answer_idx", "")).upper()
    predicted = extract_choice(solution_str)
    correct = predicted == expected and bool(expected)
    return _finalize(
        {
            "score": float(correct),
            "acc": float(correct),
            "success": True,
            "predicted_answer": predicted or "",
        }
    )
