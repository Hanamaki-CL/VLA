"""A recorded prepare/train/evaluate workflow using the same public CLI stages."""

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from .storage import read_json, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_pipeline(config_path, output_dir=None):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    project = (config_path.parent / config.get("project_root", "..")).resolve()
    root = Path(output_dir).resolve() if output_dir else (project / config["output_dir"]).resolve()
    data = read_json(config_path.parent / config["data_config"])
    training = read_json(config_path.parent / config["train_config"])
    evaluation = read_json(config_path.parent / config["eval_config"])
    if (data["suite_name"] != evaluation["suite"] or
            data["task_ids"] != evaluation["task_ids"]):
        raise ValueError("Pipeline data and evaluation must use the same suite and task IDs")
    dataset = (project / data["output"]).resolve()
    if root == dataset or dataset in root.parents or root in dataset.parents:
        raise ValueError("Pipeline output and dataset directories cannot overlap")
    data["output"] = str(dataset)
    training.update(dataset_root=str(dataset / "train"), output_dir=str(root / "train"))
    # Hub model identifiers stay unchanged; explicit local paths are resolved.
    base_model = training["base_model"]
    if base_model.startswith(("./", "../")) or Path(base_model).is_absolute():
        training["base_model"] = str((project / base_model).resolve())
    evaluation.update(output_dir=str(root / "eval"),
                      checkpoint=str(root / "train" / "checkpoints" / "last" / "pretrained_model"))
    return {"source_config": str(config_path), "project_root": str(project),
            "output_dir": str(root), "data": data, "train": training, "evaluation": evaluation}


def validate_reused_data(config, check_revision=True):
    root = Path(config["output"])
    manifest = read_json(root / "manifest.json")
    expected = {"status": "complete", "source": config["repo_id"],
                "suite": config["suite_name"], "task_ids": config["task_ids"],
                "seed": config.get("seed", 42)}
    if check_revision and config.get("revision") is not None:
        expected["revision"] = config["revision"]
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Existing dataset differs at {key}; select a new export path")
    for split in ("train", "validation", "test"):
        for relative in ("meta/stats.json", "meta/info.json", "source_episodes.json"):
            if not (root / split / relative).is_file():
                raise ValueError(f"Incomplete exported dataset: {split}/{relative}")
    return manifest


def run_stage(command, cwd, log_path):
    """Record child output; on interruption stop the Linux process group."""
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    with Path(log_path).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_pipeline(config_path, output_dir=None, reuse_data=False):
    from .cli import provenance

    plan = build_pipeline(config_path, output_dir)
    root, dataset = Path(plan["output_dir"]), Path(plan["data"]["output"])
    if reuse_data:
        validate_reused_data(plan["data"])
    elif dataset.exists():
        raise FileExistsError("Dataset already exists; use --reuse-data after checking its manifest")
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    (root / "logs").mkdir()
    for name in ("data", "train", "evaluation"):
        write_json(root / "configs" / f"{name}.json", plan[name])
    manifest = {"status": "running", "started_utc": utc_now(), "plan": plan,
                "provenance": provenance(), "stages": []}
    manifest_path = root / "pipeline.json"
    write_json(manifest_path, manifest)
    active = None
    try:
        for name, command_name, config_name in (
                ("prepare", "prepare", "data"), ("train", "train", "train"),
                ("evaluate", "evaluate", "evaluation")):
            active = {"name": name, "status": "running", "started_utc": utc_now()}
            manifest["stages"].append(active)
            if name == "prepare" and reuse_data:
                active.update(status="reused", finished_utc=utc_now())
                write_json(root / "data-manifest.json", validate_reused_data(plan["data"]))
                write_json(manifest_path, manifest)
                continue
            if name == "evaluate":
                checkpoint = Path(plan["evaluation"]["checkpoint"])
                if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
                    raise FileNotFoundError(f"Trainer did not produce a usable checkpoint: {checkpoint}")
            command = [sys.executable, "-m", "mealvla", command_name, "--config",
                       str(root / "configs" / f"{config_name}.json")]
            log_path = root / "logs" / f"{name}.log"
            active.update(argv=command, log=str(log_path.relative_to(root)))
            write_json(manifest_path, manifest)
            print(f"[{name}] Log: {log_path}", flush=True)
            run_stage(command, plan["project_root"], log_path)
            active.update(status="complete", finished_utc=utc_now())
            if name == "prepare":
                # The exporter has just resolved branch/tag revisions to a commit SHA.
                write_json(root / "data-manifest.json", validate_reused_data(plan["data"], check_revision=False))
            write_json(manifest_path, manifest)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        if active is not None and active["status"] == "running":
            active.update(status="aborted", finished_utc=utc_now())
        raise
    finally:
        manifest["finished_utc"] = utc_now()
        write_json(manifest_path, manifest)
    print(f"Report: {root / 'eval' / 'report.html'}")
    return manifest
