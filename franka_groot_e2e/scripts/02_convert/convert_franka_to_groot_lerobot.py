#!/usr/bin/env python3
"""Convert Franka synthetic episodes into a GR00T-compatible LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


DEFAULT_TASK = "Pick up every blue cube and place it in the green tray. Ignore the red cubes."
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEYS = {
    "external": "observation.images.external",
    "wrist": "observation.images.wrist",
}
EPISODE_PATTERN = re.compile(r"^episode_(\d+)(?:_run_\d+)?$")


@dataclass(frozen=True)
class EpisodeData:
    source_dir: Path
    fps: float
    width: int
    height: int
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    successful: bool
    videos: dict[str, Path]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert episode_* folders produced by franka_lift_auto_parallel.py "
            "to the LeRobot v2.1 layout expected by GR00T N1.7."
        )
    )
    parser.add_argument("input_root", type=Path, help="Directory containing episode_* folders")
    parser.add_argument("output_root", type=Path, help="New or empty LeRobot dataset directory")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Language instruction stored in tasks.jsonl")
    parser.add_argument("--chunks-size", type=int, default=1000, help="Episodes per LeRobot chunk")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes whose result.json is not successful (normally excluded)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip malformed episodes instead of failing the conversion",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def index_by_sim_step(records: Iterable[dict[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, record in enumerate(records, start=1):
        if "sim_step" not in record:
            raise ValueError(f"Missing sim_step at {path}:{row_number}")
        step = int(record["sim_step"])
        if step in indexed:
            raise ValueError(f"Duplicate sim_step={step} in {path}")
        indexed[step] = record
    return indexed


def quaternion_xyzw_to_rot6d(quaternion: Iterable[float]) -> np.ndarray:
    """Match Isaac-GR00T's Franka policy: the first two rows of an XYZW rotation matrix."""
    quat = np.asarray(list(quaternion), dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError(f"Invalid quaternion: {quat.tolist()}")
    x, y, z, w = quat / norm
    return np.asarray(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        dtype=np.float32,
    )


def vector(record: dict[str, Any], key: str, size: int, source: str) -> np.ndarray:
    if key not in record:
        raise ValueError(f"Missing {key!r} in {source}")
    value = np.asarray(record[key], dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {key} shape ({size},) in {source}, got {value}")
    return value


def video_metadata(path: Path) -> tuple[int, int, int, float, str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ")
    finally:
        capture.release()
    if frames <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(
            f"Invalid video metadata for {path}: frames={frames}, size={width}x{height}, fps={fps}"
        )
    return frames, width, height, fps, codec.lower() or "h264"


def episode_sort_key(path: Path) -> tuple[int, str]:
    match = EPISODE_PATTERN.match(path.name)
    if match is None:
        return sys.maxsize, path.name
    return int(match.group(1)), path.name


def discover_episode_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise ValueError(f"Input root does not exist or is not a directory: {input_root}")
    episodes = sorted(
        (path for path in input_root.iterdir() if path.is_dir() and EPISODE_PATTERN.match(path.name)),
        key=episode_sort_key,
    )
    if not episodes:
        raise ValueError(f"No episode_* directories found under {input_root}")
    return episodes


def load_episode(episode_dir: Path, include_failed: bool) -> EpisodeData | None:
    logs = episode_dir / "logs"
    result = read_json(logs / "result.json")
    successful = bool(result.get("completed")) and not bool(result.get("failed"))
    if not successful and not include_failed:
        print(f"[skip] {episode_dir.name}: result.json is not successful")
        return None

    scenario = read_json(logs / "scenario.json")
    timing = scenario.get("timing", {})
    args = scenario.get("args", {})
    fps = float(timing.get("sensor_fps", args.get("fps", 15.0)))
    width = int(args.get("width", 640))
    height = int(args.get("height", 360))
    if fps <= 0.0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera metadata in {logs / 'scenario.json'}")

    actions_path = logs / "actions.jsonl"
    states_path = logs / "states.jsonl"
    frames_path = logs / "frames.jsonl"
    actions = index_by_sim_step(read_jsonl(actions_path), actions_path)
    states = index_by_sim_step(read_jsonl(states_path), states_path)
    frames = index_by_sim_step(read_jsonl(frames_path), frames_path)
    action_steps, state_steps, frame_steps = set(actions), set(states), set(frames)
    if action_steps != state_steps or action_steps != frame_steps:
        raise ValueError(
            f"sim_step mismatch in {episode_dir}: "
            f"actions={len(action_steps)}, states={len(state_steps)}, frames={len(frame_steps)}, "
            f"actions-only={sorted(action_steps - state_steps - frame_steps)[:8]}, "
            f"states-only={sorted(state_steps - action_steps)[:8]}, "
            f"frames-only={sorted(frame_steps - action_steps)[:8]}"
        )
    steps = sorted(action_steps)
    if not steps:
        raise ValueError(f"No aligned samples in {episode_dir}")

    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    timestamp_values: list[float] = []
    for step in steps:
        state_record = states[step]
        action_record = actions[step]
        position = vector(state_record, "ee_pos_env", 3, f"{states_path}:sim_step={step}")
        rotation = quaternion_xyzw_to_rot6d(
            vector(state_record, "ee_quat_xyzw", 4, f"{states_path}:sim_step={step}")
        )
        gripper = float(state_record.get("gripper_width", float("nan")))
        if not math.isfinite(gripper):
            raise ValueError(f"Invalid gripper_width in {states_path}:sim_step={step}")
        state_values.append(np.concatenate((position, rotation, np.asarray([gripper], dtype=np.float32))))
        action_values.append(vector(action_record, "action", 7, f"{actions_path}:sim_step={step}"))
        timestamp = float(action_record.get("sim_time", step / fps))
        if not math.isfinite(timestamp):
            raise ValueError(f"Invalid sim_time in {actions_path}:sim_step={step}")
        timestamp_values.append(timestamp)

    timestamps = np.asarray(timestamp_values, dtype=np.float64)
    timestamps -= timestamps[0]
    if len(timestamps) > 1:
        expected_dt = 1.0 / fps
        observed_dt = np.diff(timestamps)
        if not np.allclose(observed_dt, expected_dt, rtol=0.02, atol=1.0e-4):
            raise ValueError(
                f"Non-uniform timestamps in {episode_dir}; expected dt={expected_dt:.6f}, "
                f"range=[{observed_dt.min():.6f}, {observed_dt.max():.6f}]"
            )

    videos = {name: episode_dir / f"{name}_rgb.mp4" for name in VIDEO_KEYS}
    for camera_name, video_path in videos.items():
        if not video_path.is_file():
            raise ValueError(f"Missing {camera_name} RGB video: {video_path}")
        frame_count, video_width, video_height, video_fps, _ = video_metadata(video_path)
        if frame_count != len(steps):
            raise ValueError(
                f"Frame count mismatch for {video_path}: video={frame_count}, logs={len(steps)}"
            )
        if (video_width, video_height) != (width, height):
            raise ValueError(
                f"Resolution mismatch for {video_path}: video={video_width}x{video_height}, "
                f"scenario={width}x{height}"
            )
        if not math.isclose(video_fps, fps, rel_tol=0.02, abs_tol=0.05):
            raise ValueError(f"FPS mismatch for {video_path}: video={video_fps}, scenario={fps}")

    return EpisodeData(
        source_dir=episode_dir,
        fps=fps,
        width=width,
        height=height,
        states=np.stack(state_values).astype(np.float32),
        actions=np.stack(action_values).astype(np.float32),
        timestamps=timestamps,
        successful=successful,
        videos=videos,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def stats_for(values: np.ndarray) -> dict[str, list[float]]:
    values64 = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values64, axis=0).tolist(),
        "max": np.max(values64, axis=0).tolist(),
        "mean": np.mean(values64, axis=0).tolist(),
        "std": np.std(values64, axis=0).tolist(),
        "q01": np.quantile(values64, 0.01, axis=0).tolist(),
        "q99": np.quantile(values64, 0.99, axis=0).tolist(),
    }


def prepare_output(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"Output path is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise ValueError(f"Output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir()


def parquet_frame(episode: EpisodeData, episode_index: int, global_index: int) -> pd.DataFrame:
    length = episode.length
    successful_reward = 1.0 if episode.successful else 0.0
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = successful_reward
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    return pd.DataFrame(
        {
            "observation.state": list(episode.states),
            "action": list(episode.actions),
            "timestamp": episode.timestamps,
            "annotation.human.action.task_description": np.zeros(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "next.reward": reward,
            "next.done": done,
        }
    )


def video_feature(height: int, width: int, fps: float, codec: str) -> dict[str, Any]:
    normalized_codec = "h264" if codec in {"avc1", "h264", "x264"} else codec
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.width": width,
            "video.height": height,
            "video.fps": fps,
            "video.codec": normalized_codec,
            "video.pix_fmt": "yuv420p",
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def convert(args: argparse.Namespace) -> None:
    if args.chunks_size <= 0:
        raise ValueError("--chunks-size must be positive")
    if not args.task.strip():
        raise ValueError("--task cannot be empty")

    episode_dirs = discover_episode_dirs(args.input_root)
    episodes: list[EpisodeData] = []
    failures: list[str] = []
    for episode_dir in episode_dirs:
        try:
            episode = load_episode(episode_dir, args.include_failed)
        except ValueError as exc:
            if not args.allow_incomplete:
                raise
            failures.append(str(exc))
            print(f"[skip] {episode_dir.name}: {exc}")
            continue
        if episode is not None:
            episodes.append(episode)
    if not episodes:
        raise ValueError("No usable episodes remain after validation")

    reference = episodes[0]
    for episode in episodes[1:]:
        if not math.isclose(episode.fps, reference.fps, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"Mixed episode FPS values: {reference.fps} and {episode.fps}")
        if (episode.width, episode.height) != (reference.width, reference.height):
            raise ValueError("Mixed camera resolutions are not supported")

    prepare_output(args.output_root)
    meta_dir = args.output_root / "meta"
    episode_records: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_index = 0
    codecs: dict[str, str] = {}

    for episode_index, episode in enumerate(episodes):
        chunk = episode_index // args.chunks_size
        data_relpath = DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
        data_path = args.output_root / data_relpath
        data_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_frame(episode, episode_index, global_index).to_parquet(data_path, index=False)

        for camera_name, source_video in episode.videos.items():
            video_key = VIDEO_KEYS[camera_name]
            video_relpath = VIDEO_PATH.format(
                episode_chunk=chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            destination = args.output_root / video_relpath
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, destination)
            _, _, _, _, codec = video_metadata(destination)
            codecs.setdefault(camera_name, codec)

        episode_records.append(
            {
                "episode_index": episode_index,
                "tasks": [args.task],
                "length": episode.length,
            }
        )
        all_states.append(episode.states)
        all_actions.append(episode.actions)
        global_index += episode.length
        print(f"[ok] {episode.source_dir.name} -> episode_{episode_index:06d} ({episode.length} frames)")

    total_chunks = (len(episodes) + args.chunks_size - 1) // args.chunks_size
    features: dict[str, Any] = {
        VIDEO_KEYS["external"]: video_feature(
            reference.height, reference.width, reference.fps, codecs["external"]
        ),
        VIDEO_KEYS["wrist"]: video_feature(reference.height, reference.width, reference.fps, codecs["wrist"]),
        "observation.state": {
            "dtype": "float32",
            "shape": [10],
            "names": [
                "eef_x",
                "eef_y",
                "eef_z",
                "rot6d_r0c0",
                "rot6d_r0c1",
                "rot6d_r0c2",
                "rot6d_r1c0",
                "rot6d_r1c1",
                "rot6d_r1c2",
                "gripper_width",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_panda",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": 1,
        "total_videos": len(episodes) * len(VIDEO_KEYS),
        "total_chunks": total_chunks,
        "chunks_size": args.chunks_size,
        "fps": reference.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }
    modality = {
        "state": {
            "eef_pose": {"start": 0, "end": 9, "original_key": "observation.state"},
            "gripper": {"start": 9, "end": 10, "original_key": "observation.state"},
        },
        "action": {
            "eef_delta": {"start": 0, "end": 6, "original_key": "action"},
            "gripper": {"start": 6, "end": 7, "original_key": "action"},
        },
        "video": {
            "external": {"original_key": VIDEO_KEYS["external"]},
            "wrist": {"original_key": VIDEO_KEYS["wrist"]},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    stats = {
        "observation.state": stats_for(np.concatenate(all_states, axis=0)),
        "action": stats_for(np.concatenate(all_actions, axis=0)),
    }

    write_json(meta_dir / "info.json", info)
    write_json(meta_dir / "modality.json", modality)
    write_json(meta_dir / "stats.json", stats)
    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
    write_jsonl(meta_dir / "episodes.jsonl", episode_records)

    print(
        f"Converted {len(episodes)} episodes / {global_index} frames to {args.output_root} "
        f"({len(failures)} malformed episodes skipped)"
    )


def main() -> int:
    try:
        convert(parse_args())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
