# Franka synthetic data → GR00T → IsaacLab-Arena

This directory is the reproducible, maximum-two-blue-cube workflow for Franka
synthetic-data generation, GR00T N1.7 SFT, and IsaacLab-Arena evaluation. The
production launcher uses eight GPUs and evaluates from the fixed default robot
pose.

## Validated result

The final Arena run used 100 independent episodes per task, generation-matched
camera/object/tray/lighting settings, distinct evaluation seeds, and the final
18,000-step EMA checkpoint.

| Task | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| 1 blue cube | 95 | 100 | 95% |
| **2 blue cubes** | **62** | **100** | **62%** |

The delivery metric is the **2-blue-cube success rate: 62/100 (62%)**.

- Checkpoint: `/workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-partial-progress-1200-ema/checkpoint-18000-ema`
- W&B: [franka-blue-cube-partial-progress-1200-ema](https://wandb.ai/nv-default-onboard/franka-gr00t/runs/94hk6yln)
- Machine-readable results: [assets/arena/summary.json](assets/arena/summary.json)

## Repository layout

```text
franka_groot_e2e/
├── install_franka_groot_e2e.sh
├── run_pipeline.sh
├── scripts/
│   ├── 01_generate/
│   │   ├── franka_lift_auto_parallel.py
│   │   └── analyze_franka_trajectories.py
│   └── 02_convert/
│       └── convert_franka_to_groot_lerobot.py
└── assets/
    ├── analysis/
    ├── arena/
    ├── attention/
    └── generation/
```

Source branches:

- `jihyeonRyu/Isaac-GR00T:jryu/franka-demo`
- `jihyeonRyu/IsaacLab-Arena:jryu/franka-demo`
- `jihyeonRyu/IsaacLab-Scripts:main`

## 1. Install

Use the same NVIDIA Isaac Lab Docker image for generation and Arena. The host
must expose eight CUDA GPUs and provide roughly 200 GB of free storage.

```bash
bash /workspace/IsaacLab-Scripts/franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /workspace \
  --accept-eula
```

Customer-defined paths are supported:

```bash
bash franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /customer/workspace \
  --scripts-repo /customer/workspace/IsaacLab-Scripts \
  --groot-repo /customer/workspace/Isaac-GR00T \
  --arena-repo /customer/workspace/IsaacLab-Arena \
  --models-root /customer/workspace/models
```

The installer creates isolated environments:

- `env_isaaclab`: Isaac Lab generation and Arena workers;
- `Isaac-GR00T/.venv`: conversion, SFT, inference, attention;
- `IsaacLab-Arena/.venv`: Arena coordinator;
- `.tools/ffmpeg-7`: TorchCodec-compatible FFmpeg.

It also downloads the public GR00T N1.7 3B and Cosmos Reason2 2B weights. The
model downloads do not require Hugging Face authentication. W&B is separate:

```bash
/workspace/Isaac-GR00T/.venv/bin/wandb login
```

Use `--print-config` to inspect resolved paths without installing anything.

## 2. Run end to end

```bash
cd /workspace/IsaacLab-Scripts
nohup bash franka_groot_e2e/run_pipeline.sh \
  --workspace-root /workspace \
  > /workspace/output/franka_final_pipeline.log 2>&1 &
echo $! > /workspace/output/franka_final_pipeline.pid
```

Monitor it with:

```bash
tail -f /workspace/output/franka_final_pipeline.log
cat /workspace/output/franka_e2e_pipeline_final/status.log
```

To monitor all stages and automatically package, document, commit, and push the
final evidence after successful completion:

```bash
nohup bash /workspace/IsaacLab-Scripts/franka_groot_e2e/monitor_and_finalize.sh \
  >/dev/null 2>&1 &
tail -f /workspace/output/franka_final_monitor.log
```

The launcher is restart-safe at completed stage boundaries and refuses to
silently overwrite non-empty partial outputs. Its stages are generation,
analysis, LeRobot conversion, coverage planning, SFT, final EMA attention,
maximum-two-cube Arena evaluation, and checkpoint cleanup.

## 3. Synthetic generation

The final launcher records 1,200 attempts with seed `90007`, eight GPUs, and
four vector environments per GPU. It is configured for one or two blue cubes.

- 15 FPS, 320×256 RGB, `external` and `wrist` cameras;
- stratified 4×6 target grid over X `0.33–0.70 m`, Y `-0.34–0.34 m`;
- radial workspace limit `0.68 m`;
- randomized start EEF X `0.36–0.70 m`, Y `-0.34–0.34 m`, Z `0.25–0.55 m`;
- validated floor-facing tool orientation preserved;
- full-range random background RGB plus camera/tray/object/light randomization;
- post-yaw X/Y cube-center alignment before descent;
- 10% pre-grasp recovery waypoint probability at radius 4–8 cm;
- one solver-recovery retry per cube;
- 2-cube curriculum: 75% full start and 25% with one cube preplaced;
- only successful trajectories are converted.

Maximum-two-cube analysis of the delivered generation run:

| Scenario | Successful | Attempts | Generator success rate |
|---|---:|---:|---:|
| 1 cube | 406 | 423 | 95.98% |
| 2 cubes | 334 | 388 | 86.08% |
| **Combined** | **740** | **811** | **91.25%** |

For two cubes, full-start trajectories were 245/295 (83.05%); one-preplaced
continuations were 89/93 (95.70%).

Representative videos:

- [2-cube full start](assets/generation/2c-full-start-success-episode-000266-external.mp4)
- [2-cube one-preplaced continuation](assets/generation/2c-1-preplaced-success-episode-000300-external.mp4)

Analysis:

- [trajectory distribution](assets/analysis/trajectory_distribution.png)
- [trajectory by cube count](assets/analysis/trajectory_by_blue_cube_count.png)
- [workspace coverage](assets/analysis/workspace_coverage.png)
- [scenario statistics](assets/analysis/scenario_statistics.png)
- [progress-stage statistics](assets/analysis/progress_stage_statistics.png)
- [failure causes](assets/analysis/failure_analysis.png)

## 4. LeRobot data contract

The delivered dataset has 1,068 successful episodes, 542,788 frames, two RGB
camera streams, and 15 FPS.

| Field | Contract |
|---|---|
| Images | current-frame `external` and `wrist`, 256×320 |
| State | absolute EEF XYZ + rotation 6D + gripper width, 10D |
| Action | stored delta XYZ + delta rotvec + absolute gripper command, 7D |
| Horizon | 40 frames, about 2.67 s |
| Language | `annotation.human.action.task_description` |
| Normalization | percentile min-max |

Each target is the stored `action[t:t+40]` window. The conversion does not
re-integrate or replace the recorded delta action. Progress metadata remains in
`meta/episodes.jsonl` for auditable continuation sampling.

## 5. GR00T SFT

Validated training configuration:

| Setting | Value |
|---|---|
| GPUs / global batch | 8 / 128 (16 per GPU) |
| Steps | 18,000 |
| Valid windows | 501,136 |
| Nominal data passes | 4.60 |
| LR / schedule | `1e-4` / cosine, 5% warmup |
| Weight decay | `1e-5` |
| Crop fraction | `0.98` |
| Color jitter | brightness `0.25`, contrast `0.25`, saturation `0.30`, hue `0.03` |
| State dropout | action head `0.2`, processor `0.0` |
| EMA | FP32, decay `0.999`, every optimizer step |
| Final train loss | `0.04540097` |
| Runtime | 2 h 49 min 58 s |

The final four attention probes are two 2-cube full-start samples and two
2-cube one-preplaced continuation samples, all rendered from the EMA checkpoint
at frame 120:

- [full start, episode 0](assets/attention/final-ema-episode-0-step-120.png)
- [continuation, episode 2](assets/attention/final-ema-episode-2-step-120.png)
- [full start, episode 3](assets/attention/final-ema-episode-3-step-120.png)
- [continuation, episode 6](assets/attention/final-ema-episode-6-step-120.png)

## 6. IsaacLab-Arena evaluation

Arena runs one GR00T server and one simulator worker per GPU. The launcher
selects only the 1- and 2-cube tasks, with 100 episodes each. Evaluation seeds
start at `10007` and `20007`, separate from generation seed `90007`.

The policy start pose is always the fixed default Franka pose. Randomized starts
require explicit opt-in and are not used in the reported result. Camera,
objects, tray, lighting, rendering, observation/action adapters, and task prompt
match generation/training. GR00T predicts a 40-frame action horizon and Arena
executes the first 16 actions at 15 Hz before the next inference call.

Representative Arena videos:

| Task | Success | Failure |
|---|---|---|
| 1 cube | [video](assets/arena/1-cube-success-external.mp4) | [video](assets/arena/1-cube-failure-external.mp4) |
| 2 cubes | [video](assets/arena/2-cubes-success-external.mp4) | [video](assets/arena/2-cubes-failure-external.mp4) |

To serve the packaged result assets:

```bash
python3 -m http.server 8000 \
  --directory /workspace/IsaacLab-Scripts/franka_groot_e2e/assets/arena
```

Then open `http://SERVER_IP:8000/`.
