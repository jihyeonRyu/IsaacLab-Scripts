#!/usr/bin/env python3
"""Analyze measured Franka EEF trajectories and success by blue-cube count.

Example:

    python analyze_franka_trajectories.py \
      /workspace/output/franka_parallel_dataset \
      --output-dir /workspace/output/franka_parallel_dataset/analysis

The script reads ``logs/states.jsonl`` (measured ``ee_pos_env``), not integrated
actions. It writes overall and per-scenario trajectory figures plus episode- and
scenario-level CSV/JSON statistics plus failure-cause diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    progress_stage: int
    num_blue_total: int
    num_preplaced: int
    num_remaining: int
    preplaced_blue_cube_names: list[str]
    recovery_planned: bool
    recovery_augmented: bool
    recovery_completed: bool
    solver_recovery_attempts: int
    solver_recovery_total_attempts: int
    failure_reason: str | None
    failed_state: str | None
    slot_index: int
    positions: np.ndarray
    sim_times: np.ndarray
    blue_cube_positions: np.ndarray
    blue_cube_strata: list[int | None]
    blue_cube_preplaced: list[bool]
    tray_position: np.ndarray
    workspace_bounds: tuple[float, float, float, float]
    target_workspace_bins: tuple[int, int]


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
        "--max-blue-cubes",
        type=int,
        default=None,
        help="Include only scenarios at or below this blue-cube count.",
    )
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


def load_automation_details(
    path: Path, failed: bool
) -> tuple[str | None, str | None, int]:
    if not path.is_file():
        return ("unknown", "unknown", 0) if failed else (None, None, 0)

    sequence_failed: dict[str, Any] | None = None
    total_solver_attempts = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if event.get("event") == "solver_recovery_started":
                total_solver_attempts += 1
            elif event.get("event") == "sequence_failed":
                sequence_failed = event
    if not failed:
        return None, None, total_solver_attempts
    if sequence_failed is None:
        return "unknown", "unknown", total_solver_attempts

    reason = str(sequence_failed.get("reason") or "unknown")
    failed_state = sequence_failed.get("failed_state")
    if not failed_state:
        failed_state = {
            "grasp_verification": "verify_grasp",
            "placement_verification": "retreat",
        }.get(reason, "unknown")
    return reason, str(failed_state), total_solver_attempts


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
    try:
        blue_cube_positions = np.asarray(
            [[float(value) for value in cube["pos"]] for cube in blue_cubes],
            dtype=np.float64,
        )
        if blue_cube_positions.shape != (len(blue_cubes), 3):
            raise ValueError(f"unexpected blue-cube position shape {blue_cube_positions.shape}")
        tray_position = np.asarray(
            [float(value) for value in scenario["tray_pos"]], dtype=np.float64
        )
        if tray_position.shape != (3,):
            raise ValueError(f"unexpected tray position shape {tray_position.shape}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid spawn positions in {logs_dir / 'scenario.json'}: {exc}") from exc
    scenario_args = scenario_wrapper.get("args", {})
    if not isinstance(scenario_args, dict):
        scenario_args = {}
    workspace_bounds = (
        float(scenario_args.get("workspace_x_min", 0.33)),
        float(scenario_args.get("workspace_x_max", 0.70)),
        float(scenario_args.get("workspace_y_min", -0.34)),
        float(scenario_args.get("workspace_y_max", 0.34)),
    )
    bins_value = scenario_args.get("target_workspace_bins", (4, 6))
    if not isinstance(bins_value, (list, tuple)) or len(bins_value) != 2:
        bins_value = (4, 6)
    target_workspace_bins = (int(bins_value[0]), int(bins_value[1]))
    blue_cube_preplaced = [bool(cube.get("preplaced", False)) for cube in blue_cubes]
    inferred_preplaced_cubes = sorted(
        (
            (int(cube.get("initial_placement_slot", -1)), index, cube)
            for index, cube in enumerate(blue_cubes)
            if blue_cube_preplaced[index]
        ),
        key=lambda entry: (entry[0], entry[1]),
    )
    inferred_preplaced_slots = [slot for slot, _, _ in inferred_preplaced_cubes]
    inferred_preplaced_names = [
        str(cube.get("name", f"blue_cube_{index}"))
        for _, index, cube in inferred_preplaced_cubes
    ]
    inferred_remaining_names = [
        str(cube.get("name", f"blue_cube_{index}"))
        for index, cube in enumerate(blue_cubes)
        if not blue_cube_preplaced[index]
    ]
    preplaced_blue_cube_names = [
        str(name)
        for name in scenario.get(
            "preplaced_blue_cube_names", inferred_preplaced_names
        ) or []
    ]
    remaining_blue_cube_names = [
        str(name)
        for name in scenario.get(
            "remaining_blue_cube_names", inferred_remaining_names
        ) or []
    ]
    num_preplaced = int(scenario.get("num_preplaced", len(inferred_preplaced_names)))
    num_blue_total = int(scenario.get("num_blue_total", len(blue_cubes)))
    num_remaining = int(scenario.get("num_remaining", num_blue_total - num_preplaced))
    progress_stage = int(scenario.get("progress_stage", num_preplaced))
    if (
        num_blue_total != len(blue_cubes)
        or num_preplaced != len(inferred_preplaced_names)
        or num_remaining != len(blue_cubes) - num_preplaced
        or progress_stage != num_preplaced
        or inferred_preplaced_slots != list(range(num_preplaced))
        or preplaced_blue_cube_names != inferred_preplaced_names
        or remaining_blue_cube_names != inferred_remaining_names
        or num_remaining < 1
    ):
        raise ValueError(
            f"Invalid partial-progress metadata in {logs_dir / 'scenario.json'}"
        )
    completed = bool(result.get("completed"))
    failed = bool(result.get("failed"))
    failure_reason, failed_state, total_solver_attempts = load_automation_details(
        logs_dir / "automation_events.jsonl", failed
    )
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
        progress_stage=progress_stage,
        num_blue_total=num_blue_total,
        num_preplaced=num_preplaced,
        num_remaining=num_remaining,
        preplaced_blue_cube_names=preplaced_blue_cube_names,
        recovery_planned=any(cube.get("recovery_waypoint") is not None for cube in blue_cubes),
        recovery_augmented=bool(result.get("recovery_augmented")),
        recovery_completed=bool(result.get("recovery_completed")),
        solver_recovery_attempts=int(result.get("solver_recovery_attempts", 0)),
        solver_recovery_total_attempts=total_solver_attempts,
        failure_reason=failure_reason,
        failed_state=failed_state,
        slot_index=int(result.get("slot_index", 0)),
        positions=positions,
        sim_times=sim_times,
        blue_cube_positions=blue_cube_positions,
        blue_cube_strata=[
            int(cube["position_stratum"])
            if cube.get("position_stratum") is not None
            else None
            for cube in blue_cubes
        ],
        blue_cube_preplaced=blue_cube_preplaced,
        tray_position=tray_position,
        workspace_bounds=workspace_bounds,
        target_workspace_bins=target_workspace_bins,
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
        "progress_stage": episode.progress_stage,
        "num_blue_total": episode.num_blue_total,
        "num_preplaced": episode.num_preplaced,
        "num_remaining": episode.num_remaining,
        "preplaced_blue_cube_names": ";".join(episode.preplaced_blue_cube_names),
        "recovery_planned": episode.recovery_planned,
        "recovery_augmented": episode.recovery_augmented,
        "recovery_completed": episode.recovery_completed,
        "solver_recovery_attempts": episode.solver_recovery_attempts,
        "solver_recovery_total_attempts": episode.solver_recovery_total_attempts,
        "failure_reason": episode.failure_reason or "",
        "failed_state": episode.failed_state or "",
        "slot_index": episode.slot_index,
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


def aggregate_progress_metrics(
    episodes: list[EpisodeTrajectory],
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate outcomes by total cube count and initial completed progress."""
    rows_by_episode = {row["episode_id"]: row for row in episode_rows}
    grouped: dict[tuple[int, int], list[EpisodeTrajectory]] = {}
    for episode in episodes:
        grouped.setdefault(
            (episode.blue_cube_count, episode.num_preplaced), []
        ).append(episode)

    progress_rows: list[dict[str, Any]] = []
    for (blue_cube_count, num_preplaced), group in sorted(grouped.items()):
        group_rows = [rows_by_episode[episode.episode_id] for episode in group]
        successful = sum(episode.successful for episode in group)
        progress_rows.append(
            {
                "blue_cube_count": blue_cube_count,
                "progress_stage": num_preplaced,
                "num_preplaced": num_preplaced,
                "num_remaining": blue_cube_count - num_preplaced,
                "episodes": len(group),
                "successful_episodes": successful,
                "failed_episodes": len(group) - successful,
                "success_rate_pct": 100.0 * successful / len(group),
                "trajectory_points": int(
                    sum(len(episode.positions) for episode in group)
                ),
                "mean_frames": float(
                    np.mean([row["frames"] for row in group_rows])
                ),
                "mean_duration_s": float(
                    np.mean([row["duration_s"] for row in group_rows])
                ),
                "mean_path_length_m": float(
                    np.mean([row["path_length_m"] for row in group_rows])
                ),
            }
        )
    return progress_rows


def aggregate_failure_analysis(
    episodes: list[EpisodeTrajectory],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    failed_episodes = [episode for episode in episodes if episode.failed]
    total_failures = len(failed_episodes)

    reason_counts = Counter(
        episode.failure_reason or "unknown" for episode in failed_episodes
    )
    state_counts = Counter(
        episode.failed_state or "unknown" for episode in failed_episodes
    )

    def count_rows(counter: Counter[str], label_key: str) -> list[dict[str, Any]]:
        return [
            {
                label_key: label,
                "count": count,
                "pct_of_failures": (
                    100.0 * count / total_failures if total_failures else 0.0
                ),
            }
            for label, count in counter.most_common()
        ]

    failure_rows = [
        {
            "episode_id": episode.episode_id,
            "blue_cube_count": episode.blue_cube_count,
            "progress_stage": episode.progress_stage,
            "num_preplaced": episode.num_preplaced,
            "num_remaining": episode.num_remaining,
            "failure_reason": episode.failure_reason or "unknown",
            "failed_state": episode.failed_state or "unknown",
            "solver_recovery_attempts": episode.solver_recovery_attempts,
            "solver_recovery_total_attempts": episode.solver_recovery_total_attempts,
            "slot_index": episode.slot_index,
        }
        for episode in failed_episodes
    ]

    attempts_grouped: dict[int, list[EpisodeTrajectory]] = {}
    for episode in episodes:
        attempts_grouped.setdefault(
            episode.solver_recovery_total_attempts, []
        ).append(episode)
    solver_rows: list[dict[str, Any]] = []
    for attempts, group in sorted(attempts_grouped.items()):
        successful = sum(episode.successful for episode in group)
        solver_rows.append(
            {
                "solver_recovery_total_attempts": attempts,
                "episodes": len(group),
                "successful_episodes": successful,
                "failed_episodes": len(group) - successful,
                "success_rate_pct": 100.0 * successful / len(group),
            }
        )

    by_scenario = Counter(
        (episode.blue_cube_count, episode.failed_state or "unknown")
        for episode in failed_episodes
    )
    state_by_scenario_rows = [
        {
            "blue_cube_count": blue_cube_count,
            "failed_state": failed_state,
            "count": count,
        }
        for (blue_cube_count, failed_state), count in sorted(by_scenario.items())
    ]

    analysis = {
        "total_failures": total_failures,
        "failure_reason_counts": count_rows(reason_counts, "failure_reason"),
        "failed_state_counts": count_rows(state_counts, "failed_state"),
        "solver_recovery_outcomes": solver_rows,
        "failed_states_by_blue_cube_count": state_by_scenario_rows,
    }
    return analysis, failure_rows, solver_rows


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


def remaining_blue_target_positions(episode: EpisodeTrajectory) -> np.ndarray:
    """Return only blue targets that were loose and actionable at episode start."""
    mask = np.logical_not(np.asarray(episode.blue_cube_preplaced, dtype=np.bool_))
    positions = episode.blue_cube_positions[mask]
    if len(positions) != episode.num_remaining or len(positions) < 1:
        raise ValueError(f"Invalid remaining-target mask for {episode.episode_id}")
    return positions


def workspace_coverage_rows(
    episodes: list[EpisodeTrajectory],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-target spawn rows and grid-occupancy statistics by scenario."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for target_index, (position, stratum, preplaced) in enumerate(
            zip(
                episode.blue_cube_positions,
                episode.blue_cube_strata,
                episode.blue_cube_preplaced,
                strict=True,
            )
        ):
            if preplaced:
                continue
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "blue_cube_count": episode.blue_cube_count,
                    "progress_stage": episode.progress_stage,
                    "num_remaining": episode.num_remaining,
                    "target_index": target_index,
                    "successful": episode.successful,
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                    "position_stratum": "" if stratum is None else stratum,
                    "tray_x_m": float(episode.tray_position[0]),
                    "tray_y_m": float(episode.tray_position[1]),
                }
            )

    summary: dict[str, Any] = {}
    for blue_count in sorted({episode.blue_cube_count for episode in episodes}):
        group = [episode for episode in episodes if episode.blue_cube_count == blue_count]
        points = np.concatenate(
            [remaining_blue_target_positions(episode)[:, :2] for episode in group],
            axis=0,
        )
        first_points = np.asarray(
            [remaining_blue_target_positions(episode)[0, :2] for episode in group]
        )
        x_min = min(episode.workspace_bounds[0] for episode in group)
        x_max = max(episode.workspace_bounds[1] for episode in group)
        y_min = min(episode.workspace_bounds[2] for episode in group)
        y_max = max(episode.workspace_bounds[3] for episode in group)
        x_bins = max(episode.target_workspace_bins[0] for episode in group)
        y_bins = max(episode.target_workspace_bins[1] for episode in group)
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        def grid_stats(values: np.ndarray) -> dict[str, Any]:
            counts, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=(x_edges, y_edges))
            flat = counts.astype(np.int64).reshape(-1)
            mean = float(flat.mean())
            return {
                "samples": int(len(values)),
                "occupied_cells": int(np.count_nonzero(flat)),
                "total_cells": int(len(flat)),
                "cell_count_cv": float(flat.std() / mean) if mean else 0.0,
                "cell_counts": counts.astype(np.int64).tolist(),
                "x_min_m": float(values[:, 0].min()),
                "x_max_m": float(values[:, 0].max()),
                "y_min_m": float(values[:, 1].min()),
                "y_max_m": float(values[:, 1].max()),
            }

        summary[str(blue_count)] = {
            "workspace_bounds": [x_min, x_max, y_min, y_max],
            "workspace_bins": [x_bins, y_bins],
            "all_targets": grid_stats(points),
            "first_target": grid_stats(first_points),
        }
    return rows, summary


def plot_workspace_coverage(
    episodes: list[EpisodeTrajectory],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot blue-cube spawn scatter and occupancy heatmap for every cube count."""
    cube_counts = sorted({episode.blue_cube_count for episode in episodes})
    figure, axes = plt.subplots(
        len(cube_counts),
        2,
        figsize=(12, max(4.0, 4.2 * len(cube_counts))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, blue_count in enumerate(cube_counts):
        group = [episode for episode in episodes if episode.blue_cube_count == blue_count]
        points = np.concatenate(
            [remaining_blue_target_positions(episode)[:, :2] for episode in group],
            axis=0,
        )
        first_points = np.asarray(
            [remaining_blue_target_positions(episode)[0, :2] for episode in group]
        )
        trays = np.asarray([episode.tray_position[:2] for episode in group])
        x_min = min(episode.workspace_bounds[0] for episode in group)
        x_max = max(episode.workspace_bounds[1] for episode in group)
        y_min = min(episode.workspace_bounds[2] for episode in group)
        y_max = max(episode.workspace_bounds[3] for episode in group)
        x_bins = max(episode.target_workspace_bins[0] for episode in group)
        y_bins = max(episode.target_workspace_bins[1] for episode in group)
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)
        counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(x_edges, y_edges))

        scatter_axis = axes[row_index, 0]
        scatter_axis.scatter(points[:, 0], points[:, 1], s=10, alpha=0.35, color="#2369bd", label="loose blue targets")
        scatter_axis.scatter(first_points[:, 0], first_points[:, 1], s=13, alpha=0.55, color="#0b2f59", label="first remaining target")
        scatter_axis.scatter(trays[:, 0], trays[:, 1], s=26, marker="x", color="#2a9d52", label="tray center")
        scatter_axis.set_xlim(x_min, x_max)
        scatter_axis.set_ylim(y_min, y_max)
        configure_axis(scatter_axis, "Spawn X (m)", "Spawn Y (m)")
        scatter_axis.set_title(f"{blue_count} blue cube(s): initial loose-target positions")
        scatter_axis.legend(fontsize=7, loc="best")

        heat_axis = axes[row_index, 1]
        image = heat_axis.imshow(
            counts.T,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            aspect="auto",
            cmap="Blues",
        )
        for x_index in range(x_bins):
            for y_index in range(y_bins):
                heat_axis.text(
                    0.5 * (x_edges[x_index] + x_edges[x_index + 1]),
                    0.5 * (y_edges[y_index] + y_edges[y_index + 1]),
                    str(int(counts[x_index, y_index])),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        heat_axis.set_xlabel("Spawn X (m)")
        heat_axis.set_ylabel("Spawn Y (m)")
        heat_axis.set_title(f"{x_bins}×{y_bins} workspace occupancy")
        figure.colorbar(image, ax=heat_axis, shrink=0.8, label="Blue targets")
    figure.suptitle("Blue-cube workspace coverage by scenario", fontsize=15)
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


def plot_progress_stage_statistics(
    progress_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    """Plot counts and outcomes for standard versus partial-progress starts."""
    labels = [
        f"{row['blue_cube_count']}c/{row['num_preplaced']}p"
        for row in progress_rows
    ]
    x_values = np.arange(len(labels))
    successes = np.asarray([row["successful_episodes"] for row in progress_rows])
    failures = np.asarray([row["failed_episodes"] for row in progress_rows])
    rates = np.asarray([row["success_rate_pct"] for row in progress_rows])
    lengths = np.asarray([row["mean_path_length_m"] for row in progress_rows])
    frames = np.asarray([row["mean_frames"] for row in progress_rows])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
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
        axis.set_xticks(x_values, labels, rotation=35, ha="right")
        axis.set_xlabel("Total cubes / preplaced cubes")
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Progress-stage statistics", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_failure_analysis(
    failure_analysis: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    def horizontal_counts(
        axis: Any,
        rows: list[dict[str, Any]],
        label_key: str,
        title: str,
        color: str,
    ) -> None:
        labels = [str(row[label_key]) for row in rows]
        counts = np.asarray([int(row["count"]) for row in rows])
        if not labels:
            axis.text(0.5, 0.5, "No failures", ha="center", va="center")
            axis.set_axis_off()
            return
        y_values = np.arange(len(labels))
        axis.barh(y_values, counts, color=color)
        axis.set_yticks(y_values, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Failed episodes")
        axis.set_title(title)
        axis.grid(True, axis="x", alpha=0.25)
        for y_value, count in zip(y_values, counts):
            axis.text(count + max(counts) * 0.015, y_value, str(count), va="center")

    horizontal_counts(
        axes[0, 0],
        failure_analysis["failure_reason_counts"],
        "failure_reason",
        "Terminal failure reasons",
        "#c94c4c",
    )
    horizontal_counts(
        axes[0, 1],
        failure_analysis["failed_state_counts"],
        "failed_state",
        "FSM states where failure occurred",
        "#d1843c",
    )

    solver_rows = failure_analysis["solver_recovery_outcomes"]
    attempt_labels = [str(row["solver_recovery_total_attempts"]) for row in solver_rows]
    attempt_x = np.arange(len(attempt_labels))
    successes = np.asarray([row["successful_episodes"] for row in solver_rows])
    failures = np.asarray([row["failed_episodes"] for row in solver_rows])
    axes[1, 0].bar(attempt_x, successes, label="success", color="#3a9d5d")
    axes[1, 0].bar(
        attempt_x, failures, bottom=successes, label="failed", color="#d9534f"
    )
    axes[1, 0].set_xticks(attempt_x, attempt_labels)
    axes[1, 0].set_xlabel("Total solver recovery attempts in episode")
    axes[1, 0].set_ylabel("Episodes")
    axes[1, 0].set_title("Outcome by solver-recovery usage")
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    for index, row in enumerate(solver_rows):
        axes[1, 0].text(
            index,
            row["episodes"] + max(1, max(successes + failures)) * 0.015,
            f"{row['success_rate_pct']:.1f}%",
            ha="center",
            fontsize=9,
        )

    scenario_rows = failure_analysis["failed_states_by_blue_cube_count"]
    cube_counts = sorted({int(row["blue_cube_count"]) for row in scenario_rows})
    failed_states = [
        row["failed_state"] for row in failure_analysis["failed_state_counts"]
    ]
    scenario_x = np.arange(len(cube_counts))
    bottoms = np.zeros(len(cube_counts), dtype=np.int64)
    palette = plt.get_cmap("tab10")
    for state_index, failed_state in enumerate(failed_states):
        values = np.asarray(
            [
                next(
                    (
                        int(row["count"])
                        for row in scenario_rows
                        if int(row["blue_cube_count"]) == cube_count
                        and row["failed_state"] == failed_state
                    ),
                    0,
                )
                for cube_count in cube_counts
            ]
        )
        if not values.any():
            continue
        axes[1, 1].bar(
            scenario_x,
            values,
            bottom=bottoms,
            label=failed_state,
            color=palette(state_index % 10),
        )
        bottoms += values
    axes[1, 1].set_xticks(scenario_x, [str(count) for count in cube_counts])
    axes[1, 1].set_xlabel("Number of blue cubes")
    axes[1, 1].set_ylabel("Failed episodes")
    axes[1, 1].set_title("Failure states by scenario")
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    if failed_states:
        axes[1, 1].legend(fontsize=8, ncol=2)

    figure.suptitle(
        f"Failure-cause analysis — {failure_analysis['total_failures']} failed episodes",
        fontsize=15,
    )
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
    if args.max_blue_cubes is not None and args.max_blue_cubes < 1:
        raise ValueError("--max-blue-cubes must be >= 1")

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
    if args.max_blue_cubes is not None:
        episodes = [
            episode for episode in episodes
            if episode.blue_cube_count <= args.max_blue_cubes
        ]
    if not episodes:
        raise ValueError("No valid episodes remain after validation and cube-count filtering")

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = [trajectory_metrics(episode) for episode in episodes]
    scenario_rows = aggregate_metrics(episodes, episode_rows)
    progress_rows = aggregate_progress_metrics(episodes, episode_rows)
    failure_analysis, failure_rows, solver_rows = aggregate_failure_analysis(episodes)
    workspace_rows, workspace_coverage = workspace_coverage_rows(episodes)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "scenario_success.csv", scenario_rows)
    write_csv(output_dir / "progress_stage_success.csv", progress_rows)
    write_csv(output_dir / "failure_causes.csv", failure_rows)
    write_csv(output_dir / "solver_recovery_outcomes.csv", solver_rows)
    write_csv(output_dir / "workspace_coverage.csv", workspace_rows)

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
        "progress_stage_statistics": progress_rows,
        "failure_analysis": failure_analysis,
        "workspace_coverage": workspace_coverage,
        "skipped_episodes": skipped,
        "outputs": {
            "trajectory_distribution": "trajectory_distribution.png",
            "trajectory_by_blue_cube_count": "trajectory_by_blue_cube_count.png",
            "scenario_statistics": "scenario_statistics.png",
            "progress_stage_statistics": "progress_stage_statistics.png",
            "failure_analysis": "failure_analysis.png",
            "failure_analysis_json": "failure_analysis.json",
            "episode_metrics": "episode_metrics.csv",
            "scenario_success": "scenario_success.csv",
            "progress_stage_success": "progress_stage_success.csv",
            "failure_causes": "failure_causes.csv",
            "solver_recovery_outcomes": "solver_recovery_outcomes.csv",
            "workspace_coverage_figure": "workspace_coverage.png",
            "workspace_coverage_csv": "workspace_coverage.csv",
        },
    }
    with (output_dir / "scenario_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (output_dir / "failure_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(failure_analysis, stream, indent=2, ensure_ascii=False)
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
    plot_workspace_coverage(
        episodes,
        output_dir / "workspace_coverage.png",
        args.dpi,
    )
    plot_progress_stage_statistics(
        progress_rows,
        output_dir / "progress_stage_statistics.png",
        args.dpi,
    )
    plot_failure_analysis(
        failure_analysis,
        output_dir / "failure_analysis.png",
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
    for row in progress_rows:
        print(
            f"  progress={row['blue_cube_count']} cubes / "
            f"{row['num_preplaced']} preplaced: n={row['episodes']}, "
            f"success={row['successful_episodes']}/{row['episodes']} "
            f"({row['success_rate_pct']:.1f}%)"
        )
    for row in failure_analysis["failure_reason_counts"]:
        print(
            f"  failure_reason={row['failure_reason']}: "
            f"n={row['count']} ({row['pct_of_failures']:.1f}% of failures)"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
