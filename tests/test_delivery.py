"""Regression coverage for run artifacts and orchestration boundaries."""

import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from mealvla.artifacts import export_report
from mealvla.pipeline import build_pipeline, run_pipeline, run_stage, validate_reused_data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_empty_aborted_run_exports_without_episode_file(tmp_path):
    write_json(tmp_path / "manifest.json", {
        "expected_episodes": 3, "mock_only": True, "status": "aborted", "config": {},
    })
    summary = export_report(tmp_path)
    assert summary["complete"] is False
    assert summary["mock_only"] is True
    assert summary["success_rate"] is None
    assert "MOCK" in (tmp_path / "report.html").read_text(encoding="utf-8")
    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_report_escapes_instructions_and_preserves_partial_status(tmp_path):
    record = {
        "environment": "simulation", "policy": "smolvla", "seed": 42,
        "suite": "libero_object", "task_id": 0, "initial_state": 0,
        "instruction": '<script>alert("x")</script>', "success": True,
        "first_attempt_success": True, "retries": 0, "elapsed_s": 1.0,
        "attempts": [{"steps": 2, "reason": "success", "inference_ms": [10], "control_ms": [2]}],
    }
    write_json(tmp_path / "manifest.json", {
        "expected_episodes": 1, "mock_only": False, "status": "aborted", "config": {},
    })
    (tmp_path / "episodes.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    summary = export_report(tmp_path)
    assert summary["complete"] is False
    assert summary["success_rate"] == 1
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert '<script>alert("x")</script>' not in html
    assert "&lt;script&gt;" in html
    assert "aborted" in html


def test_pipeline_resolves_configs_and_hands_off_checkpoint(tmp_path):
    configs = tmp_path / "configs"
    write_json(configs / "data.json", {
        "repo_id": "owner/data", "output": "data/export", "suite_name": "libero_object",
        "task_ids": [0], "seed": 42, "revision": None,
    })
    write_json(configs / "train.json", {"base_model": "owner/model"})
    write_json(configs / "eval.json", {"suite": "libero_object", "task_ids": [0]})
    write_json(configs / "pipeline.json", {
        "project_root": "..", "output_dir": "outputs/experiment",
        "data_config": "data.json", "train_config": "train.json", "eval_config": "eval.json",
    })
    plan = build_pipeline(configs / "pipeline.json")
    root = tmp_path / "outputs" / "experiment"
    assert Path(plan["data"]["output"]) == tmp_path / "data" / "export"
    assert Path(plan["train"]["dataset_root"]) == tmp_path / "data" / "export" / "train"
    assert Path(plan["evaluation"]["checkpoint"]) == root / "train" / "checkpoints" / "last" / "pretrained_model"
    assert not root.exists()  # Planning must not start a run or download data.
    data = json.loads((configs / "data.json").read_text())
    data["output"] = "outputs/experiment/train"
    write_json(configs / "data.json", data)
    with pytest.raises(ValueError, match="overlap"):
        build_pipeline(configs / "pipeline.json")
    data["output"] = "data/export"
    write_json(configs / "data.json", data)
    write_json(configs / "eval.json", {"suite": "libero_object", "task_ids": [1]})
    with pytest.raises(ValueError, match="task"):
        build_pipeline(configs / "pipeline.json")


def test_reused_dataset_rejects_another_source_or_partial_export(tmp_path):
    config = {"repo_id": "owner/data", "output": str(tmp_path),
              "suite_name": "libero_object", "task_ids": [0], "seed": 42, "revision": "abc"}
    write_json(tmp_path / "manifest.json", {"status": "complete", "source": "other/data"})
    with pytest.raises(ValueError):
        validate_reused_data(config)
    write_json(tmp_path / "manifest.json", {
        "status": "exporting", "source": "owner/data", "suite": "libero_object",
        "task_ids": [0], "seed": 42, "revision": "abc",
    })
    with pytest.raises(ValueError):
        validate_reused_data(config)


def test_stage_failure_preserves_log_and_exit_code(tmp_path):
    log = tmp_path / "stage.log"
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_stage([sys.executable, "-c", "print('stage-error'); raise SystemExit(7)"], tmp_path, log)
    assert caught.value.returncode == 7
    assert "stage-error" in log.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group interruption")
def test_interrupted_stage_allows_child_finally_to_run(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "from pathlib import Path\nimport time\n"
        "try:\n Path('ready').touch()\n while True: time.sleep(0.05)\n"
        "finally:\n Path('finalized').touch()\n", encoding="utf-8")
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import sys\nfrom mealvla.pipeline import run_stage\n"
        "run_stage([sys.executable, 'child.py'], '.', 'child.log')\n", encoding="utf-8")
    with (tmp_path / "parent.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(parent)], cwd=tmp_path,
                                   stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (tmp_path / "ready").exists()
            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
            assert (tmp_path / "finalized").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("failure", [None, "checkpoint", "interrupt"])
def test_pipeline_records_handoff_and_failed_stage(tmp_path, monkeypatch, failure):
    configs = tmp_path / "configs"
    data = {"repo_id": "owner/data", "output": "data/export", "suite_name": "libero_object",
            "task_ids": [0], "seed": 42, "revision": None}
    write_json(configs / "data.json", data)
    write_json(configs / "train.json", {"base_model": "owner/model"})
    write_json(configs / "eval.json", {"suite": "libero_object", "task_ids": [0]})
    write_json(configs / "pipeline.json", {
        "project_root": "..", "output_dir": "outputs/run", "data_config": "data.json",
        "train_config": "train.json", "eval_config": "eval.json",
    })
    dataset = tmp_path / "data" / "export"
    write_json(dataset / "manifest.json", {
        "status": "complete", "source": "owner/data", "suite": "libero_object",
        "task_ids": [0], "seed": 42, "revision": "resolved-sha",
    })
    for split in ("train", "validation", "test"):
        for relative in ("meta/stats.json", "meta/info.json", "source_episodes.json"):
            write_json(dataset / split / relative, {})

    def external_stage(command, cwd, log_path):
        config = json.loads(Path(command[-1]).read_text())
        Path(log_path).write_text("stage output", encoding="utf-8")
        if command[3] == "train":
            if failure == "interrupt":
                raise KeyboardInterrupt()
            if failure != "checkpoint":
                checkpoint = Path(config["output_dir"]) / "checkpoints" / "last" / "pretrained_model"
                write_json(checkpoint / "config.json", {})
                (checkpoint / "model.safetensors").touch()
        else:
            # Emulate the external evaluator rejecting an incorrect checkpoint handoff.
            assert (Path(config["checkpoint"]) / "model.safetensors").exists()

    monkeypatch.setattr("mealvla.pipeline.run_stage", external_stage)
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else FileNotFoundError):
            run_pipeline(configs / "pipeline.json", reuse_data=True)
    else:
        run_pipeline(configs / "pipeline.json", reuse_data=True)
    manifest = json.loads((tmp_path / "outputs" / "run" / "pipeline.json").read_text())
    assert manifest["status"] == ("aborted" if failure else "complete")
    assert manifest["stages"][0]["status"] == "reused"
    assert manifest["stages"][-1]["status"] == ("aborted" if failure else "complete")
    assert (tmp_path / "outputs" / "run" / "data-manifest.json").exists()
