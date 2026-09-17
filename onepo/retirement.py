"""Teacher Retirement primitives.

implements the strict reward-frontier
rule from the paper and the optional stable-retirement early-stop extension.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def retained_teacher_indices(teacher_rewards: Iterable[float], on_policy_rewards: Iterable[float]) -> list[int]:
    """Return teachers whose reward strictly exceeds the on-policy frontier."""
    teacher = np.asarray(list(teacher_rewards), dtype=np.float64)
    on_policy = np.asarray(list(on_policy_rewards), dtype=np.float64)
    if on_policy.size == 0:
        raise ValueError("Teacher Retirement requires at least one on-policy reward")
    if not np.all(np.isfinite(on_policy)):
        raise ValueError("on-policy rewards must be finite")
    frontier = float(on_policy.max())
    return [int(i) for i in np.flatnonzero(np.isfinite(teacher) & (teacher > frontier))]


@dataclass
class RetirementController:
    """Track generation-batch retirement rates and optional global stopping."""

    # Independent controls: -1 disables each corresponding stopping condition.
    max_teacher_steps: int = -1
    plateau_window: int = -1
    plateau_tolerance: float = 0.01
    plateau_min_retirement_rate: float = 0.9
    history: deque[float] = field(init=False)
    disabled_reason: str | None = None

    def __post_init__(self) -> None:
        if self.max_teacher_steps < -1:
            raise ValueError("max_teacher_steps must be -1 or non-negative")
        if self.plateau_window != -1 and self.plateau_window < 1:
            raise ValueError("plateau_window must be -1 or positive")
        if not np.isfinite(self.plateau_tolerance) or self.plateau_tolerance < 0:
            raise ValueError("plateau_tolerance must be finite and non-negative")
        if not 0 <= self.plateau_min_retirement_rate <= 1:
            raise ValueError("plateau_min_retirement_rate must be in [0, 1]")
        self.history = deque(maxlen=max(0, self.plateau_window))

    def teacher_enabled(self, step: int) -> bool:
        if self.disabled_reason is not None:
            return False
        if self.max_teacher_steps >= 0 and (self.max_teacher_steps == 0 or step > self.max_teacher_steps):
            self.disabled_reason = f"max_teacher_steps={self.max_teacher_steps}"
            return False
        return True

    def observe(self, retired: int, total: int) -> bool:
        """Stop when all rates in a full window are high enough and stable."""
        if self.disabled_reason is not None:
            return False
        if self.plateau_window == -1:
            return True
        if total <= 0:
            return self.disabled_reason is None
        rate = float(retired) / float(total)
        self.history.append(rate)
        if len(self.history) < self.plateau_window:
            return self.disabled_reason is None
        spread = max(self.history) - min(self.history)
        if (
            all(rate >= self.plateau_min_retirement_rate for rate in self.history)
            and spread <= self.plateau_tolerance
        ):
            self.disabled_reason = (
                f"retirement plateau: window={self.plateau_window}, "
                f"all rates >= {self.plateau_min_retirement_rate:.6f}, spread={spread:.6f}"
            )
        return self.disabled_reason is None
