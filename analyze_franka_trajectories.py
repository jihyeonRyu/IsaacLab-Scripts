#!/usr/bin/env python3
"""Analyze measured Franka EEF trajectories and success by blue-cube count.

Example:

    python analyze_franka_trajectories.py \
      /workspace/output/franka_parallel_dataset \
      --output-dir /workspace/output/franka_parallel_dataset/analysis

The script reads ``logs/states.jsonl`` (measured ``ee_pos_env``), not integrated
actions. It writes overall and per-scenario trajectory figures plus episode- and
scenario-level CSV/JSON statistics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class EpisodeTrajectory:
    episode_id: str
    episode_dir: Path
    blue_cube_count: int
    red_cube_count: int
    successful: bool
    completed: bool
    failed: bool
    scenario_mode: str
    seed: int | None
    recovery_planned: bool
    recovery_augmented: bool
    recovery_completed: bool
    solver_recovery_attempts: int
    positions: np.ndarray
    sim_times: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot measured Franka EEF trajectory coverage and aggregate success "
            "statistics by the number of blue cubes."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Generator output containing episode_*/logs/states.jsonl directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (default: INPUT_ROOT/trajectory_analysis).",
    )
    parser.add_argument(
        "--max-points-per-episode",
        type=int,
        default=1500,
        help="Maximum plotted points per trajectory; CSV/JSON metrics always use every state.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on a malformed episode instead of warning and skipping it.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def discover_episode_dirs(input_root: Path) -> list[Path]:
    episode_dirs = {
        states_path.parent.parent
        for states_path in input_root.rglob("logs/states.jsonl")
        if states_path.parent.parent.name.startswith("episode_")
    }
    return sorted(
        episode_dirs,
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def load_states(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    sim_times: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                position = record["ee_pos_env"]
                if len(position) != 3:
                    raise ValueError(f"ee_pos_env has {len(position)} values")
                xyz = [float(value) for value in position]
                if not all(math.isfinite(value) for value in xyz):
                    raise ValueError("ee_pos_env contains a non-finite value")
                sim_time = float(record.get("sim_time", len(sim_times)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            positions.append(xyz)
            sim_times.append(sim_time)
    if not positions:
        raise ValueError(f"No EEF states found in {path}")
    return np.asarray(positions, dtype=np.float64), np.asarray(sim_times, dtype=np.float64)


def load_episode(input_root: Path, episode_dir: Path) -> EpisodeTrajectory:
    logs_dir = episode_dir / "logs"
    scenario_wrapper = read_json(logs_dir / "scenario.json")
    scenario = scenario_wrapper.get("scenario", scenario_wrapper)
    if not isinstance(scenario, dict):
        raise ValueError(f"Invalid scenario object: {logs_dir / 'scenario.json'}")
    result = read_json(logs_dir / "result.json")
    blue_cubes = scenario.get("blue_cubes", [])
    red_cubes = scenario.get("red_cubes", [])
    if not isinstance(blue_cubes, list) or not isinstance(red_cubes, list):
        raise ValueError(f"Invalid cube lists: {logs_dir / 'scenario.json'}")
    positions, sim_times = load_states(logs_dir / "states.jsonl")
    completed = bool(result.get("completed"))
    failed = bool(result.get("failed"))
    return EpisodeTrajectory(
        episode_id=episode_dir.relative_to(input_root).as_posix(),
        episode_dir=episode_dir,
        blue_cube_count=len(blue_cubes),
        red_cube_count=len(red_cubes),
        successful=completed and not failed,
        completed=completed,
        failed=failed,
        scenario_mode=str(scenario.get("mode", "unknown")),
        seed=int(scenario["seed"]) if scenario.get("seed") is not None else None,
        recovery_planned=any(cube.get("recovery_waypoint") is not None for cube in blue_cubes),
        recovery_augmented=bool(result.get("recovery_augmented")),
        recovery_completed=bool(result.get("recovery_completed")),
        solver_recovery_attempts=int(result.get("solver_recovery_attempts", 0)),
        positions=positions,
        sim_times=sim_times,
    )


def trajectory_metrics(episode: EpisodeTrajectory) -> dict[str, Any]:
    positions = episode.positions
    deltas = np.diff(positions, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=1).sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    spans = maxs - mins
    duration = (
        float(episode.sim_times[-1] - episode.sim_times[0])
        if len(episode.sim_times) > 1
        else 0.0
    )
    return {
        "episode_id": episode.episode_id,
        "blue_cube_count": episode.blue_cube_count,
        "red_cube_count": episode.red_cube_count,
        "successful": episode.successful,
        "completed": episode.completed,
        "failed": episode.failed,
        "scenario_mode": episode.scenario_mode,
        "seed": episode.seed,
        "recovery_planned": episode.recovery_planned,
        "recovery_augmented": episode.recovery_augmented,
        "recovery_completed": episode.recovery_completed,
        "solver_recovery_attempts": episode.solver_recovery_attempts,
        "frames": int(len(positions)),
        "duration_s": duration,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "x_min_m": float(mins[0]),
        "x_max_m": float(maxs[0]),
        "x_span_m": float(spans[0]),
        "y_min_m": float(mins[1]),
        "y_max_m": float(maxs[1]),
        "y_span_m": float(spans[1]),
        "z_min_m": float(mins[2]),
        "z_max_m": float(maxs[2]),
        "z_span_m": float(spans[2]),
    }


def aggregate_metrics(
    episodes: list[EpisodeTrajectory],
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_episode = {row["episode_id"]: row for row in episode_rows}
    grouped: dict[int, list[EpisodeTrajectory]] = {}
    for episode in episodes:
        grouped.setdefault(episode.blue_cube_count, []).append(episode)

    scenario_rows: list[dict[str, Any]] = []
    for blue_cube_count, group in sorted(grouped.items()):
        group_rows = [rows_by_episode[episode.episode_id] for episode in group]
        points = np.concatenate([episode.positions for episode in group], axis=0)
        successful = sum(episode.successful for episode in group)
        failed = len(group) - successful
        scenario_rows.append(
            {
                "blue_cube_count": blue_cube_count,
                "episodes": len(group),
                "successful_episodes": successful,
                "failed_episodes": failed,
                "success_rate_pct": 100.0 * successful / len(group),
                "trajectory_points": int(sum(len(episode.positions) for episode in group)),
                "mean_frames": float(np.mean([row["frames"] for row in group_rows])),
                "mean_duration_s": float(np.mean([row["duration_s"] for row in group_rows])),
                "mean_path_length_m": float(
                    np.mean([row["path_length_m"] for row in group_rows])
                ),
                "median_path_length_m": float(
                    np.median([row["path_length_m"] for row in group_rows])
                ),
                "recovery_planned_episodes": sum(
                    episode.recovery_planned for episode in group
                ),
                "recovery_completed_episodes": sum(
                    episode.recovery_completed for episode in group
                ),
                "solver_recovery_attempts": sum(
                    episode.solver_recovery_attempts for episode in group
                ),
                "x_min_m": float(points[:, 0].min()),
                "x_max_m": float(points[:, 0].max()),
                "x_span_m": float(np.ptp(points[:, 0])),
                "y_min_m": float(points[:, 1].min()),
                "y_max_m": float(points[:, 1].max()),
                "y_span_m": float(np.ptp(points[:, 1])),
                "z_min_m": float(points[:, 2].min()),
                "z_max_m": float(points[:, 2].max()),
                "z_span_m": float(np.ptp(points[:, 2])),
            }
        )
    return scenario_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_positions(positions: np.ndarray, max_points: int) -> np.ndarray:
    if len(positions) <= max_points:
        return positions
    indices = np.linspace(0, len(positions) - 1, max_points, dtype=np.int64)
    return positions[indices]


def scenario_colors(episodes: list[EpisodeTrajectory]) -> dict[int, Any]:
    counts = sorted({episode.blue_cube_count for episode in episodes})
    color_map = plt.get_cmap("tab10")
    return {count: color_map(index % 10) for index, count in enumerate(counts)}


def configure_axis(axis: Any, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="box")


def plot_overall(
    episodes: list[EpisodeTrajectory],
    output_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    colors = scenario_colors(episodes)
    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    xy_axis = figure.add_subplot(2, 2, 1)
    xz_axis = figure.add_subplot(2, 2, 2)
    yz_axis = figure.add_subplot(2, 2, 3)
    xyz_axis = figure.add_subplot(2, 2, 4, projection="3d")

    for episode in episodes:
        points = sampled_positions(episode.positions, max_points)
        color = colors[episode.blue_cube_count]
        style = "-" if episode.successful else "--"
        alpha = 0.32 if episode.successful else 0.85
        width = 0.8 if episode.successful else 1.2
        xy_axis.plot(points[:, 0], points[:, 1], style, color=color, alpha=alpha, lw=width)
        xz_axis.plot(points[:, 0], points[:, 2], style, color=color, alpha=alpha, lw=width)
        yz_axis.plot(points[:, 1], points[:, 2], style, color=color, alpha=alpha, lw=width)
        xyz_axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            style,
            color=color,
            alpha=alpha,
            lw=width,
        )

    configure_axis(xy_axis, "EEF X (m)", "EEF Y (m)")
    configure_axis(xz_axis, "EEF X (m)", "EEF Z (m)")
    configure_axis(yz_axis, "EEF Y (m)", "EEF Z (m)")
    xy_axis.set_title("Top view (X–Y)")
    xz_axis.set_title("Side view (X–Z)")
    yz_axis.set_title("Front view (Y–Z)")
    xyz_axis.set_xlabel("X (m)")
    xyz_axis.set_ylabel("Y (m)")
    xyz_axis.set_zlabel("Z (m)")
    xyz_axis.set_title("Measured EEF trajectories (3D)")

    handles = [
        Line2D([0], [0], color=colors[count], lw=2, label=f"{count} blue cube(s)")
        for count in sorted(colors)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="success"),
            Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="failed"),
        ]
    )
    xy_axis.legend(handles=handles, loc="best", fontsize=8)
    success_count = sum(episode.successful for episode in episodes)
    figure.suptitle(
        f"Franka measured EEF trajectory coverage — "
        f"{len(episodes)} episodes, {success_count}/{len(episodes)} successful",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_by_scenario(
    episodes: list[EpisodeTrajectory],
    output_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    grouped: dict[int, list[EpisodeTrajectory]] = {}
    for episode in episodes:
        grouped.setdefault(episode.blue_cube_count, []).append(episode)
    colors = scenario_colors(episodes)
    row_count = len(grouped)
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(15, max(4.0, 4.2 * row_count)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (count, group) in enumerate(sorted(grouped.items())):
        for episode in group:
            points = sampled_positions(episode.positions, max_points)
            style = "-" if episode.successful else "--"
            alpha = 0.32 if episode.successful else 0.9
            axes[row_index, 0].plot(
                points[:, 0], points[:, 1], style, color=colors[count], alpha=alpha, lw=0.8
            )
            axes[row_index, 1].plot(
                points[:, 0], points[:, 2], style, color=colors[count], alpha=alpha, lw=0.8
            )
            axes[row_index, 2].plot(
                points[:, 1], points[:, 2], style, color=colors[count], alpha=alpha, lw=0.8
            )
        success_count = sum(episode.successful for episode in group)
        rate = 100.0 * success_count / len(group)
        title_prefix = (
            f"{count} blue cube(s): n={len(group)}, "
            f"success={success_count}/{len(group)} ({rate:.1f}%)"
        )
        for column, (xlabel, ylabel, view) in enumerate(
            (
                ("EEF X (m)", "EEF Y (m)", "X–Y"),
                ("EEF X (m)", "EEF Z (m)", "X–Z"),
                ("EEF Y (m)", "EEF Z (m)", "Y–Z"),
            )
        ):
            configure_axis(axes[row_index, column], xlabel, ylabel)
            axes[row_index, column].set_title(f"{title_prefix}\n{view}")
    figure.suptitle("Trajectory distribution by blue-cube scenario", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_scenario_statistics(
    scenario_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    labels = [str(row["blue_cube_count"]) for row in scenario_rows]
    x_values = np.arange(len(labels))
    successes = np.asarray([row["successful_episodes"] for row in scenario_rows])
    failures = np.asarray([row["failed_episodes"] for row in scenario_rows])
    rates = np.asarray([row["success_rate_pct"] for row in scenario_rows])
    lengths = np.asarray([row["mean_path_length_m"] for row in scenario_rows])
    frames = np.asarray([row["mean_frames"] for row in scenario_rows])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(x_values, successes, label="success", color="#3a9d5d")
    axes[0, 0].bar(
        x_values, failures, bottom=successes, label="failed", color="#d9534f"
    )
    axes[0, 0].set_title("Episode counts")
    axes[0, 0].legend()
    axes[0, 1].bar(x_values, rates, color="#4278b8")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_title("Success rate (%)")
    axes[1, 0].bar(x_values, lengths, color="#8c68b8")
    axes[1, 0].set_title("Mean measured path length (m)")
    axes[1, 1].bar(x_values, frames, color="#d08b36")
    axes[1, 1].set_title("Mean recorded frames")
    for axis in axes.flat:
        axis.set_xticks(x_values, labels)
        axis.set_xlabel("Number of blue cubes")
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Scenario statistics by blue-cube count", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / "trajectory_analysis"
    )
    if args.max_points_per_episode < 2:
        raise ValueError("--max-points-per-episode must be >= 2")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")

    episode_dirs = discover_episode_dirs(input_root)
    if not episode_dirs:
        raise ValueError(f"No episode states found below {input_root}")

    episodes: list[EpisodeTrajectory] = []
    skipped: list[dict[str, str]] = []
    for episode_dir in episode_dirs:
        try:
            episodes.append(load_episode(input_root, episode_dir))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if args.strict:
                raise
            skipped.append(
                {
                    "episode_id": episode_dir.relative_to(input_root).as_posix(),
                    "error": str(exc),
                }
            )
            print(f"[skip] {episode_dir}: {exc}")
    if not episodes:
        raise ValueError("No valid episodes remain after validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = [trajectory_metrics(episode) for episode in episodes]
    scenario_rows = aggregate_metrics(episodes, episode_rows)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "scenario_success.csv", scenario_rows)

    total_success = sum(episode.successful for episode in episodes)
    summary = {
        "input_root": str(input_root),
        "trajectory_source": "logs/states.jsonl:ee_pos_env",
        "episodes": len(episodes),
        "successful_episodes": total_success,
        "failed_episodes": len(episodes) - total_success,
        "success_rate_pct": 100.0 * total_success / len(episodes),
        "blue_cube_counts": sorted({episode.blue_cube_count for episode in episodes}),
        "scenario_statistics": scenario_rows,
        "skipped_episodes": skipped,
        "outputs": {
            "trajectory_distribution": "trajectory_distribution.png",
            "trajectory_by_blue_cube_count": "trajectory_by_blue_cube_count.png",
            "scenario_statistics": "scenario_statistics.png",
            "episode_metrics": "episode_metrics.csv",
            "scenario_success": "scenario_success.csv",
        },
    }
    with (output_dir / "scenario_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    plot_overall(
        episodes,
        output_dir / "trajectory_distribution.png",
        args.max_points_per_episode,
        args.dpi,
    )
    plot_by_scenario(
        episodes,
        output_dir / "trajectory_by_blue_cube_count.png",
        args.max_points_per_episode,
        args.dpi,
    )
    plot_scenario_statistics(
        scenario_rows,
        output_dir / "scenario_statistics.png",
        args.dpi,
    )

    print(
        f"Analyzed {len(episodes)} episodes: "
        f"{total_success} successful, {len(episodes) - total_success} failed."
    )
    for row in scenario_rows:
        print(
            f"  blue_cubes={row['blue_cube_count']}: "
            f"n={row['episodes']}, "
            f"success={row['successful_episodes']}/{row['episodes']} "
            f"({row['success_rate_pct']:.1f}%)"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
