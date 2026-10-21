"""Build and launch the pinned upstream trainer, without shell interpolation."""

import json
from pathlib import Path
import subprocess
import sys


def build_train_command(config: dict) -> list[str]:
    dataset = Path(config["dataset_root"]).resolve()
    manifest = json.loads((dataset.parent / "manifest.json").read_text(encoding="utf-8"))
    if dataset.name != "train" or manifest.get("status") != "complete":
        raise ValueError("Training requires the complete exported train split")
    if not (dataset / "meta" / "stats.json").is_file():
        raise ValueError("Training-split normalization statistics are missing")
    for key in ("steps", "batch_size", "chunk_size", "n_action_steps"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["n_action_steps"] > config["chunk_size"]:
        raise ValueError("n_action_steps exceeds chunk_size")
    inputs = {"observation.state": {"type": "STATE", "shape": [8]},
              "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
              "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]}}
    arguments = {
        "policy.path": config["base_model"], "policy.device": "cuda",
        "policy.push_to_hub": False, "policy.input_features": inputs,
        "policy.output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "policy.empty_cameras": 0, "policy.adapt_to_pi_aloha": False,
        "policy.train_expert_only": True, "policy.freeze_vision_encoder": True,
        "policy.chunk_size": config["chunk_size"], "policy.n_action_steps": config["n_action_steps"],
        "dataset.repo_id": "mealvla/train", "dataset.root": str(dataset),
        "dataset.use_imagenet_stats": False,
        "batch_size": config["batch_size"], "steps": config["steps"],
        "num_workers": config.get("num_workers", 4), "seed": config.get("seed", 42),
        "save_freq": config.get("save_freq", 5000), "log_freq": 100,
        "output_dir": config["output_dir"], "job_name": "mealvla_smolvla",
        "wandb.enable": False,
    }
    def encode(value):
        return json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, bool)) else str(value)
    return [sys.executable, "-m", "lerobot.scripts.lerobot_train"] + [
        f"--{key}={encode(value)}" for key, value in arguments.items()]


def train(config: dict, print_only: bool = False):
    import shlex
    command = build_train_command(config)
    print(shlex.join(command))
    if print_only:
        return
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Choose a new training output path: {output}")
    # Keep provenance next to the trainer's output; don't pre-create its output directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = output.with_name(output.name + ".launch.json")
    with provenance.open("x", encoding="utf-8") as stream:
        json.dump({"config": config, "argv": command,
                   "data_manifest": json.loads((Path(config["dataset_root"]).parent /
                                                  "manifest.json").read_text(encoding="utf-8"))},
                  stream, ensure_ascii=False, indent=2)
    subprocess.run(command, check=True)
