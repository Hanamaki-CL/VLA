"""Bounded-memory, optional video capture from the policy's two camera views."""

from pathlib import Path
import time
import numpy as np


def camera_frame(observation):
    observation.validate()
    images = [observation.images[key] for key in
              ("observation.images.image", "observation.images.image2")]
    height = max(image.shape[0] for image in images)
    padded = [np.pad(image, ((0, height - image.shape[0]), (0, 0), (0, 0))) for image in images]
    return np.ascontiguousarray(np.concatenate(padded, axis=1))


class EpisodeRecorder:
    def __init__(self, directory, stem, fps):
        self.directory, self.stem, self.fps = Path(directory), stem, fps
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.attempt = None
        self.paths = []
        self.elapsed_s = 0.0

    def __call__(self, observation, attempt, step):
        import imageio.v2 as imageio

        started = time.monotonic()
        try:
            if attempt != self.attempt:
                self.close()
                path = self.directory / f"{self.stem}_attempt{attempt}.mp4"
                if path.exists():
                    raise FileExistsError(path)
                self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264")
                self.attempt = attempt
                self.paths.append(path)
            self.writer.append_data(camera_frame(observation))
        finally:
            self.elapsed_s += time.monotonic() - started

    def close(self):
        if self.writer is not None:
            writer, self.writer = self.writer, None
            writer.close()
