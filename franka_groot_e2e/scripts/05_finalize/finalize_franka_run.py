#!/usr/bin/env python3
"""Package the latest Franka run evidence and regenerate customer-facing docs."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any


VIDEO_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:-rebuild(?P<rebuild>\d+))?"
    r"-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$"
)
RESULT_PATTERN = re.compile(
    r"^episode_results(?:_rebuild(?P<rebuild>\d+))?(?:_rank\d+)?\.jsonl$"
)
CHECKPOINT_ATTENTION_PATTERN = re.compile(
    r"^checkpoint-(?P<checkpoint>\d+)-ep(?P<episode>\d+)-"
    r"step(?P<step>\d+)\.png$"
)
FINAL_ATTENTION_PATTERN = re.compile(
    r"^final-ema-episode-(?P<episode>\d+)-step-(?P<step>\d+)\.png$"
)
REPRESENTATIVE_COUNT = 3
TASK_LABELS = {
    "franka_blue_tray_1_cube": "1 blue cube",
    "franka_blue_tray_2_cubes": "2 blue cubes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--scripts-repo", type=Path)
    parser.add_argument("--raw-dataset", type=Path)
    parser.add_argument("--lerobot-dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attention-dir", type=Path)
    parser.add_argument("--arena-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--pipeline-log", type=Path)
    parser.add_argument(
        "--experiment-name",
        default="franka-blue-cube-max2-robust-2000-ema",
    )
    parser.add_argument("--generation-attempts", type=int, default=2000)
    parser.add_argument("--generation-seed", type=int, default=91007)
    parser.add_argument("--global-batch-size", type=int, default=128)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def replace_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    staged = destination.with_name(f".{destination.name}.staged")
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(source, staged)
    if destination.exists():
        shutil.rmtree(destination)
    staged.rename(destination)


def successful_generation_episode(
    raw_dataset: Path,
    *,
    blue_cubes: int,
    preplaced: int,
) -> Path:
    for episode_dir in sorted(raw_dataset.glob("episode_*")):
        logs = episode_dir / "logs"
        scenario_path = logs / "scenario.json"
        result_path = logs / "result.json"
        video_path = episode_dir / "external_rgb.mp4"
        if not (scenario_path.is_file() and result_path.is_file() and video_path.is_file()):
            continue
        scenario_payload = read_json(scenario_path)
        scenario = scenario_payload.get("scenario", scenario_payload)
        result = read_json(result_path)
        count = int(scenario.get("num_blue_total") or len(scenario.get("blue_cubes", [])))
        if (
            count == blue_cubes
            and int(scenario.get("num_preplaced", 0)) == preplaced
            and bool(result.get("completed"))
            and not bool(result.get("failed"))
        ):
            return episode_dir
    raise RuntimeError(
        f"No successful generation episode for blue_cubes={blue_cubes}, "
        f"preplaced={preplaced}"
    )


def package_generation(raw_dataset: Path, destination: Path) -> dict[str, str]:
    staged = destination.with_name(f".{destination.name}.staged")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    selections = {
        "2c-full-start-success": successful_generation_episode(
            raw_dataset, blue_cubes=2, preplaced=0
        ),
        "2c-1-preplaced-success": successful_generation_episode(
            raw_dataset, blue_cubes=2, preplaced=1
        ),
    }
    packaged: dict[str, str] = {}
    for label, episode_dir in selections.items():
        output_name = f"{label}-{episode_dir.name}-external.mp4"
        shutil.copy2(episode_dir / "external_rgb.mp4", staged / output_name)
        packaged[label] = output_name
    (staged / "selection.json").write_text(
        json.dumps(packaged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    staged.rename(destination)
    return packaged


def arena_records_and_videos(
    arena_output: Path,
) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    selected: dict[str, list[tuple[dict[str, Any], Path]]] = {
        task: [] for task in TASK_LABELS
    }
    for results_path in sorted(arena_output.rglob("episode_results*.jsonl")):
        match = RESULT_PATTERN.match(results_path.name)
        if match is None:
            continue
        rebuild = int(match.group("rebuild") or 0)
        videos: dict[tuple[int, int, str], Path] = {}
        for video_path in sorted(results_path.parent.glob("*.mp4")):
            video_match = VIDEO_PATTERN.match(video_path.name)
            if video_match is None:
                continue
            video_rebuild = int(video_match.group("rebuild") or 0)
            if video_rebuild != rebuild:
                continue
            key = (
                int(video_match.group("env")),
                int(video_match.group("episode")),
                video_match.group("camera"),
            )
            videos[key] = video_path
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task = record.get("job_name")
            if task not in selected:
                continue
            env_id = int(record["env_id"])
            episode = int(record["episode_in_env"])
            matches = [
                (camera, path)
                for (video_env, video_episode, camera), path in videos.items()
                if video_env == env_id and video_episode == episode
            ]
            if not matches:
                raise RuntimeError(
                    f"No Arena camera video for {results_path}, env={env_id}, "
                    f"episode={episode}"
                )
            external = next(
                (path for camera, path in matches if "external" in camera.lower()),
                matches[0][1],
            )
            enriched = dict(record)
            enriched["_source_results"] = str(results_path.relative_to(arena_output))
            selected[task].append((enriched, external))
    return selected


def package_arena(arena_output: Path, destination: Path) -> dict[str, Any]:
    raw_summary = read_json(arena_output / "summary.json")
    if set(raw_summary) != set(TASK_LABELS):
        raise RuntimeError(f"Unexpected Arena task set: {sorted(raw_summary)}")
    summary: dict[str, dict[str, int | float]] = {}
    for task, result in raw_summary.items():
        episodes = int(result["episodes"])
        successes = int(result["successes"])
        if episodes != 100:
            raise RuntimeError(f"{task}: expected 100 episodes, found {episodes}")
        summary[task] = {
            "episodes": episodes,
            "successes": successes,
            "failures": episodes - successes,
            "success_rate": successes / episodes,
        }

    staged = destination.with_name(f".{destination.name}.staged")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    (staged / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = read_json(arena_output / "parallel_eval_manifest.json")
    manifest["evaluation_scope"] = "maximum_2_blue_cubes"
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records = arena_records_and_videos(arena_output)
    representatives: dict[str, Any] = {}
    for task, task_records in records.items():
        if len(task_records) != 100:
            raise RuntimeError(f"{task}: found {len(task_records)} video records")
        task_slug = "1-cube" if task.endswith("_1_cube") else "2-cubes"
        for outcome, desired in (("success", True), ("failure", False)):
            candidates = [
                item for item in task_records if item[0].get("success") is desired
            ]
            if len(candidates) < REPRESENTATIVE_COUNT:
                raise RuntimeError(
                    f"{task}: expected at least {REPRESENTATIVE_COUNT} {outcome} "
                    f"episodes, found {len(candidates)}"
                )
            candidates.sort(
                key=lambda item: (
                    int(item[0].get("episode_length", 0)),
                    str(item[0].get("_source_results", "")),
                    int(item[0].get("episode_in_env", 0)),
                    str(item[1]),
                )
            )
            chosen: list[tuple[dict[str, Any], Path]] = []
            seen_ranks: set[str] = set()
            for item in candidates:
                rank = str(item[0].get("_source_results", "")).split("/", 1)[0]
                if rank in seen_ranks:
                    continue
                chosen.append(item)
                seen_ranks.add(rank)
                if len(chosen) == REPRESENTATIVE_COUNT:
                    break
            if len(chosen) < REPRESENTATIVE_COUNT:
                chosen_paths = {path for _, path in chosen}
                for item in candidates:
                    if item[1] in chosen_paths:
                        continue
                    chosen.append(item)
                    chosen_paths.add(item[1])
                    if len(chosen) == REPRESENTATIVE_COUNT:
                        break
            if len(chosen) != REPRESENTATIVE_COUNT:
                raise RuntimeError(
                    f"{task}: selected {len(chosen)} {outcome} representatives"
                )
            for index, (record, video_path) in enumerate(chosen, start=1):
                output_name = (
                    f"{task_slug}-{outcome}-{index:02d}-external.mp4"
                )
                shutil.copy2(video_path, staged / output_name)
                representatives[output_name] = {
                    key: record.get(key)
                    for key in (
                        "job_name",
                        "success",
                        "seed",
                        "env_id",
                        "episode_in_env",
                        "episode_length",
                        "_source_results",
                    )
                }
    (staged / "representative_episodes.json").write_text(
        json.dumps(representatives, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staged / "index.html").write_text(
        arena_index_html(summary),
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    staged.rename(destination)
    return summary


def arena_index_html(summary: dict[str, dict[str, int | float]]) -> str:
    rows = []
    for task, label in TASK_LABELS.items():
        result = summary[task]
        slug = "1-cube" if task.endswith("_1_cube") else "2-cubes"
        success_videos = "".join(
            f"<video controls muted preload='metadata' "
            f"src='{slug}-success-{index:02d}-external.mp4'></video>"
            for index in range(1, REPRESENTATIVE_COUNT + 1)
        )
        failure_videos = "".join(
            f"<video controls muted preload='metadata' "
            f"src='{slug}-failure-{index:02d}-external.mp4'></video>"
            for index in range(1, REPRESENTATIVE_COUNT + 1)
        )
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{result['successes']}/{result['episodes']} "
            f"({100.0 * float(result['success_rate']):.1f}%)</td>"
            f"<td><div class='videos'>{success_videos}</div></td>"
            f"<td><div class='videos'>{failure_videos}</div></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Franka GR00T Arena results</title>
<style>
body{{font:16px system-ui;margin:2rem;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #555;padding:.7rem}}
video{{width:min(26vw,360px)}}.videos{{display:grid;gap:.6rem}}a{{color:#8cf}}
</style></head><body>
<h1>Franka GR00T — fixed-default-pose Arena evaluation</h1>
<table><thead><tr><th>Task</th><th>Result</th><th>Success</th><th>Failure</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p><a href="summary.json">summary.json</a> ·
<a href="manifest.json">manifest.json</a> ·
<a href="representative_episodes.json">representative_episodes.json</a></p>
</body></html>
"""


def package_attention(attention_dir: Path, destination: Path) -> list[str]:
    checkpoint_groups: dict[int, list[tuple[int, int, Path]]] = {}
    for png in attention_dir.glob("checkpoint-*-ep*-step*.png"):
        match = CHECKPOINT_ATTENTION_PATTERN.match(png.name)
        if match is None:
            continue
        checkpoint_step = int(match.group("checkpoint"))
        episode = int(match.group("episode"))
        frame_step = int(match.group("step"))
        require_file(png.with_suffix(".json"))
        checkpoint_groups.setdefault(checkpoint_step, []).append(
            (episode, frame_step, png)
        )
    valid_first_steps = sorted(
        step for step, probes in checkpoint_groups.items() if len(probes) == 4
    )
    if not valid_first_steps:
        raise RuntimeError("No complete four-sample checkpoint attention set found")
    first_step = valid_first_steps[0]
    first_probes = checkpoint_groups[first_step]

    final_probes: list[tuple[int, int, Path]] = []
    for png in attention_dir.glob("final-ema-episode-*-step-*.png"):
        match = FINAL_ATTENTION_PATTERN.match(png.name)
        if match is None:
            continue
        episode = int(match.group("episode"))
        frame_step = int(match.group("step"))
        require_file(png.with_suffix(".json"))
        final_probes.append((episode, frame_step, png))
    if len(final_probes) != 4:
        raise RuntimeError(
            f"Expected four final EMA attention probes, found {len(final_probes)}"
        )
    first_samples = {(episode, frame_step) for episode, frame_step, _ in first_probes}
    final_samples = {(episode, frame_step) for episode, frame_step, _ in final_probes}
    if first_samples != final_samples:
        raise RuntimeError(
            f"First/final attention samples differ: {first_samples} vs {final_samples}"
        )

    staged = destination.with_name(f".{destination.name}.staged")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    names = []
    for _, _, png in sorted(first_probes) + sorted(final_probes):
        metadata = png.with_suffix(".json")
        shutil.copy2(png, staged / png.name)
        shutil.copy2(metadata, staged / metadata.name)
        names.append(png.name)
    if destination.exists():
        shutil.rmtree(destination)
    staged.rename(destination)
    return names


def stage_duration(status_log: Path, start_text: str, end_text: str) -> str | None:
    if not status_log.is_file():
        return None
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    pattern = re.compile(r"^\[(?P<time>[^\]]+)\] (?P<message>.+)$")
    for line in status_log.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        timestamp = dt.datetime.fromisoformat(match.group("time").replace("Z", "+00:00"))
        message = match.group("message")
        if start is None and start_text in message:
            start = timestamp
        if start is not None and end_text in message:
            end = timestamp
            break
    if start is None or end is None:
        return None
    seconds = int((end - start).total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def value(mapping: dict[str, Any], *names: str, default: Any = "n/a") -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def render_readme(
    *,
    analysis: dict[str, Any],
    dataset_info: dict[str, Any],
    coverage: dict[str, Any],
    arena: dict[str, dict[str, int | float]],
    generation_videos: dict[str, str],
    attention_names: list[str],
    checkpoint: Path,
    experiment_name: str,
    attempts: int,
    generation_seed: int,
    global_batch_size: int,
    training_runtime: str | None,
) -> str:
    scenario_rows = {
        int(row["blue_cube_count"]): row for row in analysis["scenario_statistics"]
    }
    progress_rows = {
        (int(row["blue_cube_count"]), int(row["num_preplaced"])): row
        for row in analysis["progress_stage_statistics"]
    }
    one = scenario_rows[1]
    two = scenario_rows[2]
    two_full = progress_rows[(2, 0)]
    two_partial = progress_rows[(2, 1)]
    arena_one = arena["franka_blue_tray_1_cube"]
    arena_two = arena["franka_blue_tray_2_cubes"]
    max_steps = int(read_json(checkpoint / "ema_config.json")["source_step"])
    nominal_passes = value(
        coverage,
        "nominal_data_passes",
        "nominal_passes",
        default="n/a",
    )
    valid_windows = value(
        coverage,
        "valid_training_windows",
        "valid_windows",
        default="n/a",
    )
    training_runtime_text = training_runtime or "recorded in pipeline status log"
    attention_groups = []
    for prefix, title in (
        ("checkpoint-", "First saved training checkpoint"),
        ("final-ema-", "Final EMA checkpoint"),
    ):
        names = [name for name in attention_names if name.startswith(prefix)]
        if len(names) != 4:
            raise RuntimeError(f"Expected four {title} probes, found {names}")
        links = "\n".join(
            f"- [{name.removesuffix('.png')}](assets/attention/{name})"
            for name in names
        )
        attention_groups.append(f"### {title}\n\n{links}")
    attention_lines = "\n\n".join(attention_groups)

    def arena_video_links(task_slug: str, outcome: str) -> str:
        return " ".join(
            f"[{index}](assets/arena/{task_slug}-{outcome}-{index:02d}-external.mp4)"
            for index in range(1, REPRESENTATIVE_COUNT + 1)
        )
    return f"""# Franka synthetic data → GR00T → IsaacLab-Arena

This is the reproducible maximum-two-blue-cube workflow for Franka synthetic
generation, GR00T N1.7 SFT with EMA, and fixed-default-pose IsaacLab-Arena
evaluation.

## Latest validated result

The final Arena run uses 100 independent episodes per task, eight GPUs,
generation-matched camera/object/tray/lighting settings, distinct evaluation
seeds, and checkpoint `{checkpoint}`.

| Task | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| 1 blue cube | {arena_one['successes']} | {arena_one['episodes']} | {100.0 * float(arena_one['success_rate']):.1f}% |
| **2 blue cubes** | **{arena_two['successes']}** | **{arena_two['episodes']}** | **{100.0 * float(arena_two['success_rate']):.1f}%** |

Machine-readable results: [assets/arena/summary.json](assets/arena/summary.json).

## Install

Use the NVIDIA Isaac Lab container with eight CUDA GPUs:

```bash
bash /workspace/IsaacLab-Scripts/franka_groot_e2e/install_franka_groot_e2e.sh \\
  --workspace-root /workspace \\
  --accept-eula
```

The installer accepts custom `--scripts-repo`, `--groot-repo`, `--arena-repo`,
and `--models-root` paths. It creates isolated Isaac Lab, GR00T, and Arena
environments and downloads public GR00T N1.7 3B and Cosmos Reason2 2B weights.
W&B authentication remains explicit:

```bash
/workspace/Isaac-GR00T/.venv/bin/wandb login
```

## Run end to end

```bash
nohup bash /workspace/IsaacLab-Scripts/franka_groot_e2e/run_pipeline.sh \\
  --workspace-root /workspace \\
  > /workspace/output/franka_final_pipeline.log 2>&1 &
```

The restart-safe stages are generation, analysis, LeRobot conversion, coverage
planning, SFT, final EMA attention, maximum-two-cube Arena evaluation, final
evidence packaging, and checkpoint cleanup. Progress is written to
`/workspace/output/franka_e2e_pipeline_final/status.log`.

For individually runnable commands, validation checkpoints, and restart guidance,
follow [STEP_BY_STEP.md](STEP_BY_STEP.md). SFT uses the maintained Isaac-GR00T
branch and evaluation uses the maintained IsaacLab-Arena branch; their source is
not duplicated in this repository.

## Synthetic generation

- attempts / seed: `{attempts}` / `{generation_seed}`;
- 8 GPUs × 4 vector environments, 15 FPS, 320×256 RGB;
- `external` and `wrist` cameras;
- one/two cube mix 25/75%;
- two-cube one-preplaced continuation probability 30%;
- stratified target grid 4×6 and start grid 4×6×3;
- start EEF X 0.36–0.70 m, Y -0.34–0.34 m, Z 0.25–0.55 m;
- 2c1p start XY radius 0–5 cm, clearance 12–20 cm, yaw -45°–45°;
- unreachable samples resolved to safe IK-boundary poses before recording;
- 10% pre-grasp near-cube recovery, radius 4–8 cm;
- one solver recovery retry; no post-grasp wandering;
- only successful trajectories enter LeRobot.

| Scenario | Successful | Attempts | Generator success rate |
|---|---:|---:|---:|
| 1 cube | {one['successful_episodes']} | {one['episodes']} | {float(one['success_rate_pct']):.2f}% |
| 2 cubes | {two['successful_episodes']} | {two['episodes']} | {float(two['success_rate_pct']):.2f}% |
| **Combined** | **{analysis['successful_episodes']}** | **{analysis['episodes']}** | **{float(analysis['success_rate_pct']):.2f}%** |

Two-cube full starts: {two_full['successful_episodes']}/{two_full['episodes']}
({float(two_full['success_rate_pct']):.2f}%); one-preplaced continuations:
{two_partial['successful_episodes']}/{two_partial['episodes']}
({float(two_partial['success_rate_pct']):.2f}%).

Representative generation videos:

- [2-cube full start](assets/generation/{generation_videos['2c-full-start-success']})
- [2-cube one-preplaced continuation](assets/generation/{generation_videos['2c-1-preplaced-success']})

Analysis:

- [trajectory distribution](assets/analysis/trajectory_distribution.png)
- [trajectory by cube count](assets/analysis/trajectory_by_blue_cube_count.png)
- [workspace coverage](assets/analysis/workspace_coverage.png)
- [progress stages](assets/analysis/progress_stage_statistics.png)
- [failure causes](assets/analysis/failure_analysis.png)

## LeRobot data contract

Dataset: {dataset_info['total_episodes']} successful episodes,
{dataset_info['total_frames']} frames at {dataset_info['fps']} FPS.

| Field | Contract |
|---|---|
| Images | current-frame `external` and `wrist`, 256×320 |
| State | absolute EEF XYZ + rotation 6D + gripper width, 10D |
| Action | stored delta XYZ + delta rotvec + absolute gripper command, 7D |
| Horizon | 40 frames, about 2.67 s |
| Language | `annotation.human.action.task_description` |
| Normalization | percentile min-max |

## GR00T SFT

| Setting | Value |
|---|---|
| Experiment | `{experiment_name}` |
| GPUs / global batch | 8 / {global_batch_size} |
| Steps | {max_steps} |
| Valid windows | {valid_windows} |
| Nominal data passes | {nominal_passes} |
| LR / schedule | `1e-4` / cosine, 5% warmup |
| Crop / state dropout | `0.98` / `0.2` |
| Color jitter | brightness `0.25`, contrast `0.25`, saturation `0.30`, hue `0.03` |
| EMA | FP32, decay `0.999`, every optimizer step |
| Runtime | {training_runtime_text} |

Attention probes compare the same four episode/frame samples at the first saved
training checkpoint and the final EMA checkpoint:

{attention_lines}

## IsaacLab-Arena evaluation

Arena runs one GR00T server and one simulator worker per GPU. It evaluates only
the 1- and 2-cube tasks, 100 episodes each, from the fixed default Franka pose.
Evaluation seeds start at 10007 and 20007, independent of generation seed
{generation_seed}. GR00T predicts 40 frames and Arena executes the first 16
actions at 15 Hz before the next inference.

| Task | Success | Failure |
|---|---|---|
| 1 cube | {arena_video_links("1-cube", "success")} | {arena_video_links("1-cube", "failure")} |
| 2 cubes | {arena_video_links("2-cubes", "success")} | {arena_video_links("2-cubes", "failure")} |

Serve the packaged result:

```bash
python3 -m http.server 8000 \\
  --directory /workspace/IsaacLab-Scripts/franka_groot_e2e/assets/arena
```
"""


def main() -> None:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    scripts_repo = (args.scripts_repo or workspace / "IsaacLab-Scripts").resolve()
    raw_dataset = (
        args.raw_dataset
        or workspace / f"output/franka_max2_robust_{args.generation_attempts}eps_seed{args.generation_seed}"
    ).resolve()
    lerobot_dataset = (
        args.lerobot_dataset
        or workspace / f"datasets/franka_max2_robust_seed{args.generation_seed}_lerobot"
    ).resolve()
    run_dir = (
        args.run_dir
        or workspace / f"Isaac-GR00T/outputs/franka-groot-sft/{args.experiment_name}"
    ).resolve()
    checkpoint = (
        args.checkpoint
        or next(iter(sorted(run_dir.glob("checkpoint-*-ema"))), None)
    )
    if checkpoint is None:
        raise RuntimeError(f"No EMA checkpoint under {run_dir}")
    checkpoint = checkpoint.resolve()
    attention_dir = (
        args.attention_dir
        or workspace / f"Isaac-GR00T/outputs/attention/{args.experiment_name}"
    ).resolve()
    arena_output = (
        args.arena_output
        or workspace
        / "IsaacLab-Arena/outputs/franka-gr00t-parallel/"
        "max2-robust-2000-ema-default-start-8gpu-100eps"
    ).resolve()
    state_dir = (
        args.state_dir or workspace / "output/franka_e2e_pipeline_final"
    ).resolve()

    analysis_dir = raw_dataset / "trajectory_analysis"
    analysis = read_json(analysis_dir / "scenario_summary.json")
    dataset_info = read_json(lerobot_dataset / "meta/info.json")
    coverage = read_json(state_dir / "frame_coverage_audit.json")
    require_file(checkpoint / "config.json")
    require_file(checkpoint / "ema_config.json")

    assets = scripts_repo / "franka_groot_e2e/assets"
    assets.mkdir(parents=True, exist_ok=True)
    replace_directory(analysis_dir, assets / "analysis")
    generation_videos = package_generation(raw_dataset, assets / "generation")
    attention_names = package_attention(attention_dir, assets / "attention")
    arena_summary = package_arena(arena_output, assets / "arena")

    training_runtime = stage_duration(
        state_dir / "status.log",
        "starting 8-GPU SFT with EMA",
        "rendering final EMA attention for the same four checkpoint probes",
    ) or stage_duration(
        state_dir / "status.log",
        "starting 8-GPU SFT with EMA",
        "rendering four continuation-aware final EMA attention samples",
    )
    readme = render_readme(
        analysis=analysis,
        dataset_info=dataset_info,
        coverage=coverage,
        arena=arena_summary,
        generation_videos=generation_videos,
        attention_names=attention_names,
        checkpoint=checkpoint,
        experiment_name=args.experiment_name,
        attempts=args.generation_attempts,
        generation_seed=args.generation_seed,
        global_batch_size=args.global_batch_size,
        training_runtime=training_runtime,
    )
    (scripts_repo / "franka_groot_e2e/README.md").write_text(
        readme, encoding="utf-8"
    )
    two = arena_summary["franka_blue_tray_2_cubes"]
    one = arena_summary["franka_blue_tray_1_cube"]
    (assets / "README.md").write_text(
        "# Latest maximum-two-cube evidence\n\n"
        "| Stage | Assets |\n"
        "|---|---|\n"
        "| Generation | two representative 2-cube success videos |\n"
        "| Analysis | trajectory, workspace, progress, and failure plots plus CSV/JSON |\n"
        "| SFT | first-checkpoint vs final-EMA attention (four matched samples each) |\n"
        "| Arena | summary plus three success and three failure videos per task |\n\n"
        f"Fixed-default-pose Arena: **2 cubes {two['successes']}/100 "
        f"({100.0 * float(two['success_rate']):.1f}%)**; "
        f"1 cube {one['successes']}/100 "
        f"({100.0 * float(one['success_rate']):.1f}%).\n",
        encoding="utf-8",
    )
    workflow = (
        "# Franka GR00T workflow\n\n"
        "The maintained customer-facing workflow is:\n\n"
        "- overview and one-command run: "
        "`/workspace/IsaacLab-Scripts/franka_groot_e2e/README.md`\n"
        "- step-by-step runbook: "
        "`/workspace/IsaacLab-Scripts/franka_groot_e2e/STEP_BY_STEP.md`\n"
        "- repository: `jihyeonRyu/IsaacLab-Scripts`, branch `main`\n\n"
        "## Latest validated scope\n\n"
        "Maximum two blue cubes; fixed default Franka evaluation start; eight GPUs; "
        "independent seeds; 100 episodes per task.\n\n"
        "| Task | Successes | Episodes | Success rate |\n"
        "|---|---:|---:|---:|\n"
        f"| 1 blue cube | {one['successes']} | 100 | "
        f"{100.0 * float(one['success_rate']):.1f}% |\n"
        f"| **2 blue cubes** | **{two['successes']}** | **100** | "
        f"**{100.0 * float(two['success_rate']):.1f}%** |\n\n"
        "Installation, generation, conversion, SFT, evaluation, schema, plots, "
        "attention maps, and representative videos are in the maintained README.\n"
    )
    (workspace / "FRANKA_GROOT_WORKFLOW.md").write_text(
        workflow, encoding="utf-8"
    )
    (state_dir / "finalization.done").write_text(
        dt.datetime.now(dt.timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis_episodes": analysis["episodes"],
                "lerobot_episodes": dataset_info["total_episodes"],
                "checkpoint": str(checkpoint),
                "arena": arena_summary,
                "assets": str(assets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
