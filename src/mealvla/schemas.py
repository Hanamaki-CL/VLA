"""Shared contracts. All control timestamps use time.monotonic()."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    images: dict[str, np.ndarray]
    instruction: str
    timestamp: float

    def validate(self) -> None:
        if self.state.shape != (8,) or not np.isfinite(self.state).all():
            raise ValueError("Expected finite LIBERO state: eef xyz + axis-angle + 2 gripper positions")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("A non-empty language instruction is required")
        if not math.isfinite(self.timestamp):
            raise ValueError("Observation timestamp must be finite")
        for name in ("observation.images.image", "observation.images.image2"):
            image = self.images.get(name)
            if image is None or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Missing HWC RGB camera: {name}")
            if image.dtype != np.uint8 or min(image.shape[:2]) < 1:
                raise ValueError(f"Expected nonempty uint8 camera: {name}")


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    observed_at: float


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    success: bool
    terminated: bool


@dataclass(frozen=True)
class ExecutionConfig:
    horizon: int = 10
    max_steps: int = 280
    max_retries: int = 0
    action_ttl_s: float = 5.0
    inference_timeout_s: float = 5.0
    control_hz: float = 20.0
    realtime: bool = False

    def __post_init__(self):
        for key in ("horizon", "max_steps", "max_retries"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.horizon < 1 or self.max_steps < 1 or self.max_retries < 0:
            raise ValueError("horizon/max_steps must be positive; retries must be nonnegative")
        for key in ("action_ttl_s", "inference_timeout_s", "control_hz"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
