import json
import pytest
from mealvla.training import build_train_command


def test_training_uses_only_exported_train_statistics(tmp_path):
    root = tmp_path / "train"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "stats.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete"}))
    config = {"dataset_root": str(root), "base_model": "lerobot/smolvla_base",
              "batch_size": 8, "steps": 100, "chunk_size": 50,
              "n_action_steps": 10, "output_dir": "output with spaces"}
    argv = build_train_command(config)
    assert f"--dataset.root={root.resolve()}" in argv
    assert "--output_dir=output with spaces" in argv
    assert "--policy.push_to_hub=false" in argv
    assert "--dataset.use_imagenet_stats=false" in argv
    config["dataset_root"] = str(tmp_path / "test")
    with pytest.raises(ValueError):
        build_train_command(config)
