"""Data, observation, and action-queue contract tests."""

import numpy as np
import pytest
from mealvla.control import ActionQueue
from mealvla.schemas import ActionChunk, ExecutionConfig
from mealvla.data import split_episodes, camera_to_hwc
from mealvla.envs import quaternion_to_axis_angle


def test_invalid_replacement_discards_old_actions():
    queue = ActionQueue(2, 1)
    queue.replace(ActionChunk(np.zeros((3, 7)), 10), 10)
    with pytest.raises(ValueError):
        queue.replace(ActionChunk(np.full((2, 7), np.nan), 10), 10)
    assert not queue


def test_only_requested_prefix_is_executed():
    queue = ActionQueue(2, 1)
    queue.replace(ActionChunk(np.arange(21).reshape(3, 7) / 100, 10), 10)
    assert queue.pop(10.1)[0] == pytest.approx(0)
    assert queue.pop(10.2)[0] == pytest.approx(0.07)
    assert not queue


@pytest.mark.parametrize("now", [9, 11, float("nan")])
def test_clock_errors_and_expiration_clear_queue(now):
    queue = ActionQueue(2, 1)
    queue.replace(ActionChunk(np.zeros((2, 7)), 10), 10)
    with pytest.raises(TimeoutError):
        queue.pop(now)
    assert not queue


@pytest.mark.parametrize("actions", [np.zeros((1, 6)), np.full((1, 7), 2), np.zeros((0, 7))])
def test_rejects_incompatible_action_space(actions):
    with pytest.raises(ValueError):
        ActionQueue(2, 1).replace(ActionChunk(actions, 1), 1)


def test_split_preserves_whole_unique_episodes_and_is_deterministic():
    source = {"a": list(range(10)), "b": list(range(10, 20))}
    splits = split_episodes(source, 42)
    assert splits == split_episodes(source, 42)
    assert len(splits["train"]) == 16
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert sorted(sum(splits.values(), [])) == list(range(20))


def test_duplicate_episode_cannot_leak_across_tasks():
    with pytest.raises(ValueError):
        split_episodes({"a": [0, 1, 2], "b": [2, 3, 4]})


def test_exported_camera_matches_hwc_metadata():
    image = np.zeros((3, 4, 5), dtype=np.float32)
    image[0] = 0.25
    image[1] = 0.5
    image[2] = 0.75
    converted = camera_to_hwc(image)
    assert converted.shape == (4, 5, 3)
    np.testing.assert_array_equal(converted[2, 3], [0.25, 0.5, 0.75])
    assert converted.flags.c_contiguous


def test_axis_angle_order_is_xyz_not_quaternion_wxyz():
    np.testing.assert_allclose(quaternion_to_axis_angle([0, 0, 0, 1]), [0, 0, 0])
    np.testing.assert_allclose(quaternion_to_axis_angle([0, 0, 1, 0]), [0, 0, np.pi])


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"max_retries": -1},
                                    {"control_hz": float("nan")}, {"max_steps": 1.5}])
def test_invalid_execution_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExecutionConfig(**kwargs)
