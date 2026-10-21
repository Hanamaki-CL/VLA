"""Episode-level split and physical export, avoiding global-statistics leakage."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import numpy as np


def split_episodes(groups: dict[str, list[int]], seed: int = 42) -> dict[str, list[int]]:
    result = {"train": [], "validation": [], "test": []}
    all_ids = [episode for values in groups.values() for episode in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Episode IDs must be unique across tasks")
    rng = random.Random(seed)
    for task in sorted(groups):
        episodes = sorted(groups[task])
        if len(episodes) < 3:
            raise ValueError(f"Need at least 3 episodes for task {task}")
        rng.shuffle(episodes)
        heldout = max(1, len(episodes) // 10)
        result["test"].extend(episodes[:heldout])
        result["validation"].extend(episodes[heldout:2 * heldout])
        result["train"].extend(episodes[2 * heldout:])
    if not all(result.values()):
        raise ValueError("Empty split")
    return {name: sorted(values) for name, values in result.items()}


def validate_frame(frame: dict):
    for key, shape in (("observation.state", (8,)), ("action", (7,))):
        array = np.asarray(frame[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}: expected finite {shape}")
    if not isinstance(frame.get("task"), str) or not frame["task"].strip():
        raise ValueError("Frame needs language task text")


def camera_to_hwc(image):
    """Make decoded CHW tensors agree with the exported HWC image metadata."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("Expected three-dimensional RGB image")
    if array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite RGB image")
    return np.ascontiguousarray(array)


def prepare_dataset(repo_id: str, output: str, suite_name: str,
                    task_ids: list[int], seed: int = 42, revision: str | None = None):
    """Export image-backed datasets. Original IDs and resolved revision are retained."""
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from libero.libero import benchmark

    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new output directory: {root}")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Supply unique task IDs")
    suite = benchmark.get_benchmark_dict()[suite_name]()
    if any(t < 0 or t >= suite.n_tasks for t in task_ids):
        raise ValueError("Task ID outside suite")
    wanted = {suite.get_task(t).language: t for t in task_ids}
    resolved = HfApi().dataset_info(repo_id, revision=revision).sha
    # LeRobot may trust existing cache files without comparing Hub revisions.
    # A SHA-specific root prevents a newer manifest from naming older cached data.
    source_cache = root.parent / ".mealvla_source_cache" / repo_id.replace("/", "--") / resolved
    source = LeRobotDataset(repo_id, revision=resolved, root=source_cache)
    # This is metadata only; images are decoded only for the selected episodes below.
    index = source.hf_dataset.select_columns(["episode_index", "task_index"])
    frame_indices, episode_tasks = defaultdict(list), {}
    for position, row in enumerate(index):
        episode, task = int(row["episode_index"]), int(row["task_index"])
        text = str(source.meta.tasks.iloc[task].name)
        if text not in wanted:
            continue
        if episode in episode_tasks and episode_tasks[episode] != text:
            raise ValueError("Multi-instruction episodes require an explicit task mapping")
        episode_tasks[episode] = text
        frame_indices[episode].append(position)
    groups = {text: sorted(ep for ep, label in episode_tasks.items() if label == text)
              for text in wanted}
    splits = split_episodes(groups, seed)
    root.mkdir(parents=True)
    manifest = {"source": repo_id, "revision": resolved, "seed": seed,
                "suite": suite_name, "task_ids": task_ids, "task_text_to_id": wanted,
                "splits": splits, "format": "LeRobot image-backed",
                "statistics": "recomputed independently by each exported dataset",
                "status": "exporting"}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_keys = [key for key in source.features if key == "action" or key.startswith("observation.")]
    features = {key: deepcopy(source.features[key]) for key in feature_keys}
    for key in source.meta.camera_keys:
        shape = tuple(source.features[key]["shape"])
        if shape[0] == 3:
            shape = (shape[1], shape[2], 3)
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Unsupported camera shape: {key} {shape}")
        features[key] = {"dtype": "image", "shape": shape,
                         "names": ["height", "width", "channels"]}
    for split, episodes in splits.items():
        target = LeRobotDataset.create(repo_id=f"mealvla/{split}", root=root / split,
                                        fps=source.fps, features=deepcopy(features),
                                        robot_type=source.meta.robot_type, use_videos=False)
        mapping = []
        try:
            for new_id, old_id in enumerate(episodes):
                previous_time = -1.0
                for expected_index, row_id in enumerate(frame_indices[old_id]):
                    row = source[row_id]
                    timestamp = float(row["timestamp"])
                    if timestamp <= previous_time or int(row["frame_index"]) != expected_index:
                        raise ValueError(f"Non-contiguous frame index/timestamp in episode {old_id}")
                    previous_time = timestamp
                    frame = {key: row[key] for key in feature_keys}
                    frame["task"] = row["task"]
                    validate_frame(frame)
                    for key in source.meta.camera_keys:
                        frame[key] = camera_to_hwc(frame[key])
                    target.add_frame(frame)
                target.save_episode()
                mapping.append({"export_episode": new_id, "source_episode": old_id})
        finally:
            target.finalize()
        (root / split / "source_episodes.json").write_text(
            json.dumps(mapping, indent=2), encoding="utf-8")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
