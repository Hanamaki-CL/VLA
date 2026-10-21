"""LIBERO adapter: camera rotation and state convention match HF LeRobot data."""

import time
import numpy as np
from .schemas import Observation, StepResult


def quaternion_to_axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected nonzero xyzw quaternion")
    q = q / np.linalg.norm(q)
    w = np.clip(q[3], -1, 1)
    den = np.sqrt(max(0.0, 1.0 - w * w))
    return np.zeros(3) if den < 1e-10 else q[:3] * (2 * np.arccos(w) / den)


class LiberoEnvironment:
    kind = "simulation"

    def __init__(self, suite_name: str, task_id: int, initial_state: int = 0):
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv, get_task_init_states

        suites = benchmark.get_benchmark_dict()
        if suite_name not in suites:
            raise ValueError(f"Unknown suite: {suite_name}")
        suite = suites[suite_name]()
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"Task index {task_id} out of range")
        if not 0 <= initial_state < len(get_task_init_states(suite, task_id)):
            raise ValueError("Initial-state index out of range; no silent modulo reuse")
        self.instruction = suite.get_task(task_id).language
        self.env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                             obs_type="pixels_agent_pos", observation_width=256,
                             observation_height=256, init_states=True,
                             episode_index=initial_state, control_mode="relative")

    def _convert(self, raw):
        robot = raw["robot_state"]
        state = np.concatenate((robot["eef"]["pos"],
                                quaternion_to_axis_angle(robot["eef"]["quat"]),
                                robot["gripper"]["qpos"])).astype(np.float32)
        # HF LIBERO preprocessing rotates both axes; NOT a single vertical flip.
        images = {f"observation.images.{key}": image[::-1, ::-1].copy()
                  for key, image in raw["pixels"].items()}
        return Observation(state, images, self.instruction, time.monotonic())

    def reset(self, seed):
        raw, _ = self.env.reset(seed=seed)
        return self._convert(raw)

    def step(self, action):
        raw, _, terminated, truncated, info = self.env.step(action)
        return StepResult(self._convert(raw), bool(info.get("is_success", False)),
                          bool(terminated or truncated))

    def close(self):
        self.env.close()


class MockEnvironment:
    """A scalar smoke fixture, not a physics simulation or a robotics benchmark."""
    kind = "mock"
    instruction = "mock transport smoke check"

    def _observation(self):
        state = np.zeros(8, dtype=np.float32)
        state[0] = self.position
        images = {key: np.zeros((32, 32, 3), dtype=np.uint8)
                  for key in ("observation.images.image", "observation.images.image2")}
        return Observation(state, images, self.instruction, time.monotonic())

    def reset(self, seed):
        self.position = 0.0
        return self._observation()

    def step(self, action):
        self.position += float(action[0])
        success = self.position >= 0.9
        return StepResult(self._observation(), success, success)

    def close(self):
        pass
