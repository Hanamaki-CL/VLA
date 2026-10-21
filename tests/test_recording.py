import numpy as np
import pytest

from mealvla.envs import MockEnvironment
from mealvla.policies import MockPolicy
from mealvla.recording import camera_frame
from mealvla.runner import run_episode
from mealvla.schemas import ExecutionConfig


def test_camera_frame_keeps_both_views_in_order():
    observation = MockEnvironment().reset(42)
    observation.images["observation.images.image"][:] = 20
    observation.images["observation.images.image2"][:] = 90
    frame = camera_frame(observation)
    assert frame.shape == (32, 64, 3)
    assert np.all(frame[:, :32] == 20)
    assert np.all(frame[:, 32:] == 90)


def test_observer_sees_each_reset_and_step_across_retries():
    seen = []
    result = run_episode(MockEnvironment(), MockPolicy(),
                         ExecutionConfig(max_steps=1, max_retries=1), 42,
                         observer=lambda obs, attempt, step: seen.append((attempt, step)))
    assert seen == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert result["retries"] == 1


def test_recording_failure_is_not_counted_as_policy_contract_failure():
    def broken_observer(*args):
        raise ValueError("encoder failed")
    with pytest.raises(RuntimeError, match="recording"):
        run_episode(MockEnvironment(), MockPolicy(), ExecutionConfig(), 42,
                    observer=broken_observer)
