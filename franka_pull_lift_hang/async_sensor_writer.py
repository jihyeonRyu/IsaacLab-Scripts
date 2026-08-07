"""Asynchronous RGB and raw instance-segmentation dataset writer."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from PIL import Image


class AsyncSensorWriter:
    """Move encoded image and array file I/O off the simulation thread."""

    def __init__(self, root: Path, camera_names, max_queue: int = 128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = Queue(maxsize=max_queue)
        self.error = None
        for camera_name in camera_names:
            (self.root / camera_name / "rgb").mkdir(parents=True, exist_ok=True)
        self.thread = Thread(target=self._run, name="sensor-dataset-writer", daemon=True)
        self.thread.start()

    def submit(self, frame_index: int, physics_step: int, cameras, sample_metadata=None) -> None:
        if self.error is not None:
            raise RuntimeError(f"async sensor writer failed: {self.error}") from self.error
        frames = {}
        for camera_name, camera in cameras.items():
            # Keep the simulation thread free of synchronous GPU-to-CPU
            # readback.  The worker owns the immutable GPU snapshots and does
            # transfer, encoding, and disk I/O.
            rgb = camera.data.output["rgb"][0, ..., :3].detach().clone()
            instance_tensor = camera.data.output.get("instance_segmentation_fast")
            instance_map = (
                instance_tensor[0].detach().clone() if instance_tensor is not None else None
            )
            labels = camera.data.info.get("instance_segmentation_fast", {}) if instance_map is not None else None
            frames[camera_name] = (rgb, instance_map, labels)
        record = {"frame": frame_index, "physics_step": physics_step}
        record.update(sample_metadata or {})
        self.queue.put((frame_index, frames, record))

    def _run(self) -> None:
        metadata_path = self.root / "frames.jsonl"
        written_labels = set()
        try:
            with metadata_path.open("a", encoding="utf-8") as metadata:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    frame_index, frames, record = item
                    stem = f"{frame_index:06d}"
                    for camera_name, (rgb, instance_map, labels) in frames.items():
                        rgb = rgb.to("cpu").numpy()
                        if instance_map is not None:
                            instance_map = instance_map.to("cpu").numpy()
                        camera_dir = self.root / camera_name
                        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(camera_dir / "rgb" / f"{stem}.png")
                        if instance_map is not None:
                            instance_dir = camera_dir / "instance_segmentation"
                            instance_dir.mkdir(parents=True, exist_ok=True)
                            np.save(instance_dir / f"{stem}.npy", np.asarray(instance_map), allow_pickle=False)
                        if instance_map is not None and camera_name not in written_labels:
                            with (camera_dir / "instance_id_labels.json").open("w", encoding="utf-8") as stream:
                                json.dump(labels, stream, ensure_ascii=False, indent=2, default=str)
                            written_labels.add(camera_name)
                    metadata.write(
                        json.dumps(
                            record, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)
                        ) + "\n"
                    )
                    metadata.flush()
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"async sensor writer failed: {self.error}") from self.error
