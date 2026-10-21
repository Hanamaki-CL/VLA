"""Fail-closed bounds and freshness checks for normalized LIBERO actions."""

from collections import deque
import math
import numpy as np
from .schemas import ActionChunk


class ActionQueue:
    def __init__(self, horizon: int, ttl_s: float):
        if not isinstance(horizon, int) or horizon < 1 or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("Invalid queue horizon or TTL")
        self.horizon, self.ttl_s = horizon, ttl_s
        self._actions = deque()
        self._observed_at = 0.0

    def clear(self):
        self._actions.clear()

    def __bool__(self):
        return bool(self._actions)

    def replace(self, chunk: ActionChunk, now: float):
        self.clear()  # Never retain an older chunk if replacement is rejected.
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
            raise ValueError("Expected a nonempty [T, 7] action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite action")
        if np.abs(actions).max() > 1.00001:
            raise ValueError("Action outside LIBERO [-1, 1]; check denormalization/control mode")
        self._observed_at = chunk.observed_at
        self._check_age(now)
        self._actions.extend(np.clip(actions[: self.horizon], -1, 1).copy())

    def _check_age(self, now: float):
        age = now - self._observed_at
        if not math.isfinite(age) or age < 0 or age >= self.ttl_s:
            self.clear()
            raise TimeoutError("Expired action chunk or mismatched clock")

    def pop(self, now: float) -> np.ndarray:
        self._check_age(now)
        if not self._actions:
            raise IndexError("Action queue is empty")
        return self._actions.popleft()
