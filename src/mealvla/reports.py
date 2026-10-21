"""Honest denominators: retries do not become independent evaluation episodes."""

from collections import Counter
import math
import numpy as np


def wilson(successes: int, total: int):
    if total < 1:
        return None
    z = 1.95996398454
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(records: list[dict], expected_episodes: int | None = None):
    kinds = {(r["environment"], r["policy"]) for r in records}
    if len(kinds) > 1:
        raise ValueError("Cannot combine different environment/policy kinds")
    seen = set()
    for record in records:
        identity = (record.get("suite"), record.get("task_id"), record.get("initial_state"), record["seed"])
        if identity in seen:
            raise ValueError("Duplicate evaluation case")
        seen.add(identity)
    count = len(records)
    successes = sum(bool(r["success"]) for r in records)
    first = sum(bool(r["first_attempt_success"]) for r in records)
    inference = [ms for r in records for a in r["attempts"] for ms in a["inference_ms"]]
    control = [ms for r in records for a in r["attempts"] for ms in a["control_ms"]]
    failed = Counter(r["attempts"][-1]["reason"] for r in records if not r["success"])
    def percentiles(values):
        return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None
    kind = next(iter(kinds)) if kinds else None
    groups = {}
    for task in sorted({str(r.get("task_id")) for r in records}):
        subset = [r for r in records if str(r.get("task_id")) == task]
        groups[task] = {"episodes": len(subset), "successes": sum(r["success"] for r in subset)}
    return {"kind": kind, "mock_only": bool(kind and "mock" in kind), "episodes": count,
            "expected_episodes": expected_episodes,
            "complete": expected_episodes is not None and count == expected_episodes,
            "successes": successes, "success_rate": successes / count if count else None,
            "success_rate_ci95": wilson(successes, count),
            "first_attempt_success_rate": first / count if count else None,
            "first_attempt_ci95": wilson(first, count),
            "reset_retries": sum(r["retries"] for r in records),
            "inference_ms": percentiles(inference), "control_step_ms": percentiles(control),
            "mean_episode_s": float(np.mean([r["elapsed_s"] for r in records])) if count else None,
            "failure_reasons": dict(failed), "per_task": groups,
            "protocol": "MealVLA custom subset; not a full official LIBERO benchmark"}
