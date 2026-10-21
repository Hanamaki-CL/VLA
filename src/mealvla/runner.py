"""Synchronous receding-horizon evaluation; reset retries are recorded separately."""

from dataclasses import asdict
import time
from .control import ActionQueue
from .schemas import ExecutionConfig


def run_episode(env, policy, config: ExecutionConfig, seed: int) -> dict:
    attempts, events = [], []
    queue = ActionQueue(config.horizon, config.action_ttl_s)
    started = time.monotonic()
    for retry in range(config.max_retries + 1):
        queue.clear()
        events.append({"state": "resetting", "attempt": retry})
        # Reset/model setup errors propagate: infrastructure failures aren't task failures.
        policy.reset()
        observation = env.reset(seed)
        observation.validate()
        latencies, control_latencies = [], []
        reason, success, steps = "max_steps", False, 0
        attempt_start = time.monotonic()
        events.append({"state": "executing", "attempt": retry})
        try:
            while steps < config.max_steps:
                tick = time.monotonic()
                if not queue:
                    infer_start = time.monotonic()
                    chunk = policy.predict(observation)
                    elapsed = time.monotonic() - infer_start
                    latencies.append(elapsed * 1000)
                    if elapsed >= config.inference_timeout_s:
                        raise TimeoutError("Inference exceeded deadline; returned actions discarded")
                    queue.replace(chunk, time.monotonic())
                action = queue.pop(time.monotonic())
                step_start = time.monotonic()
                outcome = env.step(action)
                control_latencies.append((time.monotonic() - step_start) * 1000)
                steps += 1
                observation = outcome.observation
                observation.validate()
                if outcome.success:
                    success, reason = True, "success"
                    break
                if outcome.terminated:
                    reason = "environment_terminated"
                    break
                if config.realtime:
                    time.sleep(max(0.0, 1 / config.control_hz - (time.monotonic() - tick)))
        except TimeoutError as exc:
            reason = f"deadline: {exc}"
        except ValueError as exc:
            reason = f"contract: {exc}"
        finally:
            queue.clear()
        attempts.append({"success": success, "steps": steps, "reason": reason,
                         "inference_ms": latencies, "control_ms": control_latencies,
                         "elapsed_s": time.monotonic() - attempt_start})
        if success:
            events.append({"state": "completed", "attempt": retry})
            break
        events.append({"state": "reset_retry" if retry < config.max_retries else "failed",
                       "attempt": retry})
    return {"environment": env.kind, "policy": policy.kind, "instruction": env.instruction,
            "seed": seed, "first_attempt_success": attempts[0]["success"],
            "success": any(x["success"] for x in attempts), "retries": len(attempts) - 1,
            "elapsed_s": time.monotonic() - started, "attempts": attempts,
            "events": events, "execution": asdict(config)}
