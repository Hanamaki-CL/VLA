"""SmolVLA inference and mock implementations of the action-chunk interface."""

from typing import Protocol
import numpy as np
from .schemas import ActionChunk, Observation


class Policy(Protocol):
    kind: str

    def reset(self) -> None: ...
    def predict(self, observation: Observation) -> ActionChunk: ...


class MockPolicy:
    """A fixed action generator. It has no learned visual or language ability."""
    kind = "mock"

    def reset(self):
        pass

    def predict(self, observation):
        observation.validate()
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = 0.1
        return ActionChunk(actions, observation.timestamp)


class SmolVLAPolicy:
    kind = "smolvla"

    def __init__(self, checkpoint: str, device: str = "cuda"):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as UpstreamPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch, self.device = torch, device
        self.model = UpstreamPolicy.from_pretrained(checkpoint).to(device).eval()
        cfg = self.model.config
        required = {"observation.state": (8,), "observation.images.image": None,
                    "observation.images.image2": None}
        for key, shape in required.items():
            if key not in cfg.input_features or (shape and tuple(cfg.input_features[key].shape) != shape):
                raise ValueError(f"Checkpoint lacks LIBERO feature {key}; use the fine-tuned checkpoint")
        if tuple(cfg.output_features["action"].shape) != (7,):
            raise ValueError("Checkpoint must output 7 LIBERO actions")
        self.preprocess, self.postprocess = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self):
        self.model.reset()
        self.preprocess.reset()
        self.postprocess.reset()

    def predict(self, observation):
        observation.validate()
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(observation.state.copy()).float(),
                 "task": observation.instruction}
        for name, image in observation.images.items():
            batch[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            batch = self.preprocess(batch)
            actions = self.model.predict_action_chunk(batch)
            # Postprocess each [B, D] action, using saved TRAINING statistics.
            decoded = [self.postprocess(actions[:, i, :])[0].detach().cpu().numpy()
                       for i in range(actions.shape[1])]
        # CPU copy above synchronizes GPU work, so runner latency includes actual completion.
        return ActionChunk(np.stack(decoded), observation.timestamp)
