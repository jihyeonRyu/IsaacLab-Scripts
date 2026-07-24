# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run Franka GR00T evaluation as one policy-server and Arena-worker pair per GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

TASK_BASE_SEEDS = {
    "franka_blue_tray_1_cube": 10007,
    "franka_blue_tray_2_cubes": 20007,
    "franka_blue_tray_3_cubes": 30007,
}
"""Evaluation seeds chosen outside the synthetic-data seed range."""

DEFAULT_CHECKPOINT = Path(
    "/workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-sft-crop098-aug-v2/checkpoint-10000"
)
DEFAULT_EXPERIMENT_CONFIG = Path(
    "isaaclab_arena_environments/experiment_configs/franka_blue_tray_gr00t_experiment.yaml"
)
DEFAULT_COSMOS_MODEL = Path("/workspace/models/Cosmos-Reason2-2B")
RTX_KIT_ARGS = (
    "--/rtx/hydra/progressiveSceneLoad=false "
    "--/rtx/hydra/geometrySyncLoads=true "
    "--/rtx-transient/hydra/geometrystreaming/syncLoad=true"
)


def split_episode_budget(total_episodes: int, worker_count: int) -> list[int]:
    """Split an episode budget evenly while assigning every worker at least one episode."""
    assert total_episodes > 0, "total_episodes must be positive"
    assert worker_count > 0, "worker_count must be positive"
    assert total_episodes >= worker_count, "total_episodes must be at least worker_count"
    base, remainder = divmod(total_episodes, worker_count)
    return [base + int(rank < remainder) for rank in range(worker_count)]


def build_worker_overrides(rank: int, episode_count: int, task_name: str | None = None) -> list[str]:
    """Build Hydra overrides for one single-environment, optionally single-task worker."""
    if task_name is not None:
        assert task_name in TASK_BASE_SEEDS, f"Unknown Franka task: {task_name}"
    selected_tasks = TASK_BASE_SEEDS if task_name is None else {task_name: TASK_BASE_SEEDS[task_name]}
    overrides = [f"~runs.{name}" for name in TASK_BASE_SEEDS if task_name is not None and name != task_name]
    for selected_task_name, base_seed in selected_tasks.items():
        overrides.extend([
            f"runs.{selected_task_name}.environment_builder.num_envs=1",
            f"runs.{selected_task_name}.environment_builder.seed={base_seed + rank}",
            f"runs.{selected_task_name}.rollout_limit.num_episodes={episode_count}",
        ])
    return overrides


def summarize_results(output_dir: Path, expected_episodes_per_task: int) -> dict[str, dict[str, float | int]]:
    """Aggregate worker JSONL files into task-level episode counts and success rates."""
    records_by_task: dict[str, list[dict]] = {task_name: [] for task_name in TASK_BASE_SEEDS}
    for results_path in sorted(output_dir.rglob("episode_results*.jsonl")):
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task_name = record.get("job_name")
            if task_name in records_by_task:
                records_by_task[task_name].append(record)

    summary: dict[str, dict[str, float | int]] = {}
    for task_name, records in records_by_task.items():
        assert (
            len(records) == expected_episodes_per_task
        ), f"{task_name}: expected {expected_episodes_per_task} episodes, found {len(records)}"
        successes = sum(record.get("success") is True for record in records)
        summary[task_name] = {
            "episodes": len(records),
            "successes": successes,
            "success_rate": successes / len(records),
        }
    return summary


def _default_arena_repo() -> Path:
    return Path(os.environ.get("ARENA_REPO", "/workspace/IsaacLab-Arena"))


def _build_parser() -> argparse.ArgumentParser:
    arena_repo = _default_arena_repo()
    gr00t_repo = Path(os.environ.get("GROOT_REPO", "/workspace/Isaac-GR00T"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated physical GPU IDs; defaults to 0..num_gpus-1.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--base-port", type=int, default=5655)
    parser.add_argument("--server-timeout-sec", type=float, default=900.0)
    parser.add_argument("--arena-repo", type=Path, default=arena_repo)
    parser.add_argument("--gr00t-repo", type=Path, default=gr00t_repo)
    parser.add_argument(
        "--cosmos-model-path",
        type=Path,
        default=Path(os.environ.get("GROOT_COSMOS_MODEL_PATH", DEFAULT_COSMOS_MODEL)),
    )
    parser.add_argument("--arena-python", type=Path, default=arena_repo / ".venv/bin/python")
    parser.add_argument("--gr00t-python", type=Path, default=gr00t_repo / ".venv/bin/python")
    parser.add_argument("--experiment-config", type=Path, default=arena_repo / DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--record-camera-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record observation-camera videos for validation (default: enabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan without starting processes.",
    )
    return parser


def _parse_gpu_ids(args: argparse.Namespace) -> list[int]:
    assert args.num_gpus > 0, "--num-gpus must be positive"
    if args.gpu_ids is None:
        gpu_ids = list(range(args.num_gpus))
    else:
        gpu_ids = [int(value.strip()) for value in args.gpu_ids.split(",") if value.strip()]
        assert len(gpu_ids) == args.num_gpus, "--gpu-ids count must equal --num-gpus"
        assert len(set(gpu_ids)) == len(gpu_ids), "--gpu-ids must not contain duplicates"
    assert args.episodes_per_task >= len(gpu_ids), "--episodes-per-task must be at least the worker count"
    assert 1 <= args.base_port <= 65535 - len(gpu_ids) + 1, "server port range exceeds 65535"
    return gpu_ids


def _assert_file_layout(args: argparse.Namespace) -> None:
    required_paths = [
        args.checkpoint,
        args.arena_python,
        args.gr00t_python,
        args.cosmos_model_path / "config.json",
        args.experiment_config,
        args.gr00t_repo / "gr00t/eval/run_gr00t_server.py",
        args.arena_repo / "isaaclab_arena/evaluation/experiment_runner.py",
        args.arena_repo / "isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py",
    ]
    missing = [path for path in required_paths if not path.exists()]
    assert not missing, f"Required paths are missing: {missing}"


def _assert_ports_available(ports: list[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))


def _assert_gpus_available(gpu_ids: list[int]) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    available_gpu_ids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    missing_gpu_ids = set(gpu_ids) - available_gpu_ids
    assert not missing_gpu_ids, f"Requested GPUs are not visible to nvidia-smi: {sorted(missing_gpu_ids)}"


def _command_text(gpu_id: int, command: list[str]) -> str:
    return f"CUDA_VISIBLE_DEVICES={gpu_id} {shlex.join(command)}"


def _server_command(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(args.gr00t_python),
        str(args.gr00t_repo / "gr00t/eval/run_gr00t_server.py"),
        f"--model_path={args.checkpoint}",
        "--embodiment_tag=NEW_EMBODIMENT",
        "--device=cuda:0",
        "--host=127.0.0.1",
        f"--port={port}",
    ]


def _wait_command(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(args.gr00t_python),
        str(args.arena_repo / "isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py"),
        "--host=127.0.0.1",
        f"--port={port}",
        f"--timeout-sec={args.server_timeout_sec}",
    ]


def _worker_command(
    args: argparse.Namespace,
    rank: int,
    port: int,
    episode_count: int,
    worker_output_dir: Path,
    task_name: str | None = None,
) -> list[str]:
    command = [
        str(args.arena_python),
        str(args.arena_repo / "isaaclab_arena/evaluation/experiment_runner.py"),
        "--headless",
        "--enable_cameras",
        "--device=cuda:0",
        f"--kit_args={RTX_KIT_ARGS}",
        f"--experiment_config={args.experiment_config}",
        f"--experiment_output_directory={worker_output_dir}",
        "--remote_host=127.0.0.1",
        f"--remote_port={port}",
    ]
    if args.record_camera_video:
        command.append("--record_camera_video")
    return command + build_worker_overrides(rank, episode_count, task_name)


def _process_environment(gpu_id: int, gr00t_repo: Path, cosmos_model_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["PYTHONUNBUFFERED"] = "1"
    python_path_entries = [str(gr00t_repo)]
    if environment.get("PYTHONPATH"):
        python_path_entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    environment["GROOT_COSMOS_MODEL_PATH"] = str(cosmos_model_path)
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("ACCEPT_EULA", "Y")
    environment.setdefault("PRIVACY_CONSENT", "Y")
    return environment


def _handle_termination_signal(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 20.0
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    for process in processes:
        if process.poll() is None:
            process.wait()


def _build_aggregate_report(args: argparse.Namespace, output_dir: Path) -> None:
    command = [
        str(args.arena_python),
        "-c",
        (
            "from pathlib import Path; "
            "from isaaclab_arena.visualization.report import build_report; "
            "print(build_report(Path(__import__('sys').argv[1])))"
        ),
        str(output_dir),
    ]
    subprocess.run(
        command,
        cwd=args.arena_repo,
        env=_process_environment(0, args.gr00t_repo, args.cosmos_model_path),
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Launch all policy-server and Arena-worker pairs, then aggregate their results."""
    args = _build_parser().parse_args(argv)
    gpu_ids = _parse_gpu_ids(args)
    worker_count = len(gpu_ids)
    episode_counts = split_episode_budget(args.episodes_per_task, worker_count)
    ports = [args.base_port + rank for rank in range(worker_count)]
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        output_dir = args.arena_repo / "outputs/franka-gr00t-parallel" / timestamp

    print(f"checkpoint: {args.checkpoint}")
    print(f"output: {output_dir}")
    print(f"workers: {worker_count}; GPUs: {gpu_ids}; episodes/task: {episode_counts}")
    for rank, (gpu_id, port, episode_count) in enumerate(zip(gpu_ids, ports, episode_counts, strict=True)):
        print(f"[rank {rank}] GPU {gpu_id}, port {port}, episodes/task {episode_count}")
        print(f"  server: {_command_text(gpu_id, _server_command(args, port))}")
    for task_name in TASK_BASE_SEEDS:
        task_slug = task_name.removeprefix("franka_blue_tray_")
        print(f"[task {task_name}] fresh Arena process per GPU")
        for rank, (gpu_id, port, episode_count) in enumerate(zip(gpu_ids, ports, episode_counts, strict=True)):
            worker_output_dir = output_dir / f"rank-{rank:02d}" / f"stage-{task_slug}"
            worker_command = _worker_command(args, rank, port, episode_count, worker_output_dir, task_name)
            print(f"  rank {rank}: {_command_text(gpu_id, worker_command)}")
    if args.dry_run:
        return 0

    _assert_file_layout(args)
    _assert_gpus_available(gpu_ids)
    assert not output_dir.exists() or not any(output_dir.iterdir()), f"Output directory is not empty: {output_dir}"
    _assert_ports_available(ports)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cosmos_model_path": str(args.cosmos_model_path.resolve()),
        "gpu_ids": gpu_ids,
        "ports": ports,
        "episodes_per_task": args.episodes_per_task,
        "episode_counts_by_rank": episode_counts,
        "task_base_seeds": TASK_BASE_SEEDS,
        "fresh_arena_process_per_task": True,
        "rtx_kit_args": RTX_KIT_ARGS,
        "task_seeds_by_rank": [
            {task_name: base_seed + rank for task_name, base_seed in TASK_BASE_SEEDS.items()}
            for rank in range(worker_count)
        ],
    }
    (output_dir / "parallel_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    processes: list[subprocess.Popen] = []
    server_processes: list[subprocess.Popen] = []
    log_handles = []
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_termination_signal)
    try:
        for rank, (gpu_id, port) in enumerate(zip(gpu_ids, ports, strict=True)):
            log_handle = (logs_dir / f"server-rank-{rank:02d}.log").open("w", encoding="utf-8")
            log_handles.append(log_handle)
            server = subprocess.Popen(
                _server_command(args, port),
                cwd=args.gr00t_repo,
                env=_process_environment(gpu_id, args.gr00t_repo, args.cosmos_model_path),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(server)
            server_processes.append(server)

        for rank, (gpu_id, port) in enumerate(zip(gpu_ids, ports, strict=True)):
            wait_log_path = logs_dir / f"server-wait-rank-{rank:02d}.log"
            with wait_log_path.open("w", encoding="utf-8") as wait_log:
                result = subprocess.run(
                    _wait_command(args, port),
                    cwd=args.gr00t_repo,
                    env=_process_environment(gpu_id, args.gr00t_repo, args.cosmos_model_path),
                    stdout=wait_log,
                    stderr=subprocess.STDOUT,
                )
            assert result.returncode == 0, f"GR00T server rank {rank} was not ready; see {wait_log_path}"
            assert server_processes[rank].poll() is None, f"GR00T server rank {rank} exited during startup"

        for task_name in TASK_BASE_SEEDS:
            task_worker_processes = []
            task_slug = task_name.removeprefix("franka_blue_tray_")
            print(f"Starting fresh Arena processes for task '{task_name}'", flush=True)
            for rank, (gpu_id, port, episode_count) in enumerate(zip(gpu_ids, ports, episode_counts, strict=True)):
                worker_output_dir = output_dir / f"rank-{rank:02d}" / f"stage-{task_slug}"
                # A fresh Kit process per task prevents Fabric/RTX transforms from leaking across stage rebuilds.
                worker_output_dir.mkdir(parents=True)
                log_path = logs_dir / f"arena-{task_slug}-rank-{rank:02d}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                log_handles.append(log_handle)
                worker = subprocess.Popen(
                    _worker_command(args, rank, port, episode_count, worker_output_dir, task_name),
                    cwd=worker_output_dir,
                    env=_process_environment(gpu_id, args.gr00t_repo, args.cosmos_model_path),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(worker)
                task_worker_processes.append(worker)

            while any(worker.poll() is None for worker in task_worker_processes):
                for rank, server in enumerate(server_processes):
                    assert server.poll() is None, f"GR00T server rank {rank} exited during evaluation"
                for rank, worker in enumerate(task_worker_processes):
                    if worker.poll() not in (None, 0):
                        raise RuntimeError(
                            f"Arena worker rank {rank} failed for {task_name}; "
                            f"see logs/arena-{task_slug}-rank-{rank:02d}.log"
                        )
                time.sleep(2.0)

            failed_workers = [rank for rank, worker in enumerate(task_worker_processes) if worker.returncode != 0]
            assert not failed_workers, f"Arena workers failed for {task_name}: {failed_workers}"
    finally:
        _terminate_processes(processes)
        for log_handle in log_handles:
            log_handle.close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    summary = summarize_results(output_dir, args.episodes_per_task)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_aggregate_report(args, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
