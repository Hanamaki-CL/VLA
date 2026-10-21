"""Command-line entry points. Optional ML dependencies load only when needed."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from . import __version__
from .schemas import ExecutionConfig


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def provenance():
    versions = {}
    for name in ("numpy", "torch", "lerobot", "hf-libero"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"mealvla": __version__, "python": sys.version, "packages": versions,
            "git_commit": commit, "git_dirty": dirty,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def evaluate(config, mock=False):
    from .envs import MockEnvironment, LiberoEnvironment
    from .policies import MockPolicy, SmolVLAPolicy
    from .reports import summarize
    from .runner import run_episode

    execution = ExecutionConfig(**config.get("execution", {}))
    n = config["episodes_per_task"]
    tasks = config["task_ids"]
    if not isinstance(n, int) or n < 1 or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Require positive episodes_per_task and unique task IDs")
    start = config.get("initial_state_start", 0)
    if not isinstance(start, int) or start < 0:
        raise ValueError("initial_state_start must be nonnegative")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    manifest = {"config": config, "provenance": provenance(), "status": "starting",
                "mock_only": mock, "expected_episodes": n * len(tasks)}
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        if mock:
            policy = MockPolicy()
        else:
            import torch
            policy = SmolVLAPolicy(config["checkpoint"], config.get("device", "cuda"))
        for task in tasks:
            for episode in range(n):
                initial_state = start + episode
                seed = config.get("seed", 42) + initial_state
                random.seed(seed)
                np.random.seed(seed)
                if not mock:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                env = MockEnvironment() if mock else LiberoEnvironment(config["suite"], task, initial_state)
                try:
                    record = run_episode(env, policy, execution, seed)
                finally:
                    env.close()
                record.update({"suite": config.get("suite", "mock"), "task_id": task,
                               "initial_state": initial_state, "checkpoint": config.get("checkpoint")})
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                records.append(record)
                write_json(output / "summary.json", summarize(records, n * len(tasks)))
                print(f"task={task} initial_state={initial_state} success={record['success']}")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "summary.json", summarize(records, n * len(tasks)))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    print(json.dumps(summarize(records, n * len(tasks)), indent=2))


def doctor(cuda_check=False):
    result = provenance()
    result["platform"] = sys.platform
    if cuda_check:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install the CUDA 12.8 build on the 5090 host")
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        (x @ x.T).square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(x.grad).all().item():
            raise RuntimeError("Non-finite CUDA backward result")
        result["cuda"] = {"name": torch.cuda.get_device_name(), "build": torch.version.cuda,
                          "capability": torch.cuda.get_device_capability(),
                          "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                          "tensor_forward_backward": "passed"}
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MealVLA project commands")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Inspect environment; optional CUDA operation")
    check.add_argument("--cuda-check", action="store_true")
    demo = commands.add_parser("demo", help="MOCK ONLY: check plumbing, no VLA benchmark")
    demo.add_argument("--output", default="outputs/mock-demo")
    prep = commands.add_parser("prepare", help="Export task-specific episode splits")
    prep.add_argument("--config", default="configs/data.json")
    training = commands.add_parser("train", help="Launch upstream SmolVLA fine-tuning")
    training.add_argument("--config", default="configs/train_5090.json")
    training.add_argument("--print-only", action="store_true")
    evaluation = commands.add_parser("evaluate", help="Evaluate a fine-tuned checkpoint in LIBERO")
    evaluation.add_argument("--config", default="configs/eval_libero.json")
    report = commands.add_parser("report", help="Summarize one evaluation run")
    report.add_argument("run_dir")
    tasks = commands.add_parser("tasks", help="List official LIBERO task IDs and instructions")
    tasks.add_argument("--suite", default="libero_object")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            doctor(args.cuda_check)
        elif args.command == "demo":
            evaluate({"output_dir": args.output, "task_ids": [0], "episodes_per_task": 3,
                      "execution": {"max_steps": 20}}, mock=True)
        elif args.command == "prepare":
            from .data import prepare_dataset
            print(json.dumps(prepare_dataset(**read_json(args.config)), ensure_ascii=False, indent=2))
        elif args.command == "train":
            from .training import train
            train(read_json(args.config), args.print_only)
        elif args.command == "evaluate":
            evaluate(read_json(args.config))
        elif args.command == "report":
            from .reports import summarize
            root = Path(args.run_dir)
            records = [json.loads(line) for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
            manifest = read_json(root / "manifest.json")
            print(json.dumps(summarize(records, manifest["expected_episodes"]), ensure_ascii=False, indent=2))
        elif args.command == "tasks":
            from libero.libero import benchmark
            suite = benchmark.get_benchmark_dict()[args.suite]()
            print(json.dumps([{ "task_id": i, "instruction": suite.get_task(i).language}
                              for i in range(suite.n_tasks)], ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"MealVLA: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
