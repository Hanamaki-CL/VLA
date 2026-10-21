"""Execution/report contract tests; no GPU or network needed when run later."""

import pytest
from mealvla.envs import MockEnvironment
from mealvla.policies import MockPolicy
from mealvla.runner import run_episode
from mealvla.schemas import ExecutionConfig
from mealvla.reports import summarize


def test_reset_retries_are_bounded_and_do_not_multiply_denominator():
    result = run_episode(MockEnvironment(), MockPolicy(),
                         ExecutionConfig(max_steps=1, max_retries=2), seed=42)
    assert len(result["attempts"]) == 3
    assert not result["success"]
    summary = summarize([result], expected_episodes=1)
    assert summary["episodes"] == 1
    assert summary["success_rate"] == 0
    assert summary["reset_retries"] == 2
    assert summary["mock_only"] is True


def test_success_stops_retry_loop():
    result = run_episode(MockEnvironment(), MockPolicy(),
                         ExecutionConfig(max_steps=20, max_retries=2), seed=42)
    assert result["success"]
    assert result["retries"] == 0
    assert result["events"][-1]["state"] == "completed"


def test_duplicate_case_is_rejected():
    result = run_episode(MockEnvironment(), MockPolicy(), ExecutionConfig(max_steps=1), 42)
    with pytest.raises(ValueError):
        summarize([result, result])


def test_empty_or_partial_run_does_not_claim_completion():
    result = summarize([], expected_episodes=10)
    assert result["success_rate"] is None
    assert result["complete"] is False
    assert result["success_rate_ci95"] is None
