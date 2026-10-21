import numpy as np
import pytest
import json

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


def test_encoder_close_failure_still_closes_environment(tmp_path, monkeypatch):
    from mealvla.cli import evaluate

    closed = []

    class TrackedEnvironment(MockEnvironment):
        def close(self):
            closed.append(True)

    class BrokenEncoder:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            pass

        def close(self):
            raise RuntimeError("encoder flush failed")

    monkeypatch.setattr("mealvla.envs.MockEnvironment", TrackedEnvironment)
    monkeypatch.setattr("mealvla.recording.EpisodeRecorder", BrokenEncoder)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="encoder flush"):
        evaluate({"output_dir": str(root), "task_ids": [0], "episodes_per_task": 1,
                  "record_video": True}, mock=True)
    assert closed == [True]
    assert json.loads((root / "manifest.json").read_text())["status"] == "aborted"
    assert (root / "report.html").exists()
