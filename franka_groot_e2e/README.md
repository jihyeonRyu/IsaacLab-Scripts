# Franka synthetic data → GR00T → IsaacLab-Arena

This is the reproducible maximum-two-blue-cube workflow for Franka synthetic
generation, GR00T N1.7 SFT with EMA, and fixed-default-pose IsaacLab-Arena
evaluation.

## Latest validated result

The final Arena run uses 100 independent episodes per task, eight GPUs,
generation-matched camera/object/tray/lighting settings, distinct evaluation
seeds, and checkpoint `/workspace/Isaac-GR00T/outputs/franka-groot-sft/franka-blue-cube-max2-robust-2000-ema/checkpoint-25000-ema`.

| Task | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| 1 blue cube | 97 | 100 | 97.0% |
| **2 blue cubes** | **67** | **100** | **67.0%** |

Machine-readable results: [assets/arena/summary.json](assets/arena/summary.json).

## Install

Use the NVIDIA Isaac Lab container with eight CUDA GPUs:

```bash
bash /workspace/IsaacLab-Scripts/franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /workspace \
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
nohup bash /workspace/IsaacLab-Scripts/franka_groot_e2e/run_pipeline.sh \
  --workspace-root /workspace \
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

- attempts / seed: `2000` / `91007`;
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
| 1 cube | 478 | 495 | 96.57% |
| 2 cubes | 1279 | 1505 | 84.98% |
| **Combined** | **1757** | **2000** | **87.85%** |

Two-cube full starts: 826/1032
(80.04%); one-preplaced continuations:
453/473
(95.77%).

Representative generation videos:

- [2-cube full start](assets/generation/2c-full-start-success-episode_000002-external.mp4)
- [2-cube one-preplaced continuation](assets/generation/2c-1-preplaced-success-episode_000001-external.mp4)

Analysis:

- [trajectory distribution](assets/analysis/trajectory_distribution.png)
- [trajectory by cube count](assets/analysis/trajectory_by_blue_cube_count.png)
- [workspace coverage](assets/analysis/workspace_coverage.png)
- [progress stages](assets/analysis/progress_stage_statistics.png)
- [failure causes](assets/analysis/failure_analysis.png)

## LeRobot data contract

Dataset: 1757 successful episodes,
754545 frames at 15.0 FPS.

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
| Experiment | `franka-blue-cube-max2-robust-2000-ema` |
| GPUs / global batch | 8 / 128 |
| Steps | 25000 |
| Valid windows | 686022 |
| Nominal data passes | 4.664573439335765 |
| LR / schedule | `1e-4` / cosine, 5% warmup |
| Crop / state dropout | `0.98` / `0.2` |
| Color jitter | brightness `0.25`, contrast `0.25`, saturation `0.30`, hue `0.03` |
| EMA | FP32, decay `0.999`, every optimizer step |
| Runtime | 3h 58m 14s |

Attention probes compare the same four episode/frame samples at the first saved
training checkpoint and the final EMA checkpoint:

### First saved training checkpoint

- [checkpoint-1000-ep0-step120](assets/attention/checkpoint-1000-ep0-step120.png)
- [checkpoint-1000-ep1-step120](assets/attention/checkpoint-1000-ep1-step120.png)
- [checkpoint-1000-ep2-step120](assets/attention/checkpoint-1000-ep2-step120.png)
- [checkpoint-1000-ep3-step120](assets/attention/checkpoint-1000-ep3-step120.png)

### Final EMA checkpoint

- [final-ema-episode-0-step-120](assets/attention/final-ema-episode-0-step-120.png)
- [final-ema-episode-1-step-120](assets/attention/final-ema-episode-1-step-120.png)
- [final-ema-episode-2-step-120](assets/attention/final-ema-episode-2-step-120.png)
- [final-ema-episode-3-step-120](assets/attention/final-ema-episode-3-step-120.png)

## IsaacLab-Arena evaluation

Arena runs one GR00T server and one simulator worker per GPU. It evaluates only
the 1- and 2-cube tasks, 100 episodes each, from the fixed default Franka pose.
Evaluation seeds start at 10007 and 20007, independent of generation seed
91007. GR00T predicts 40 frames and Arena executes the first 16
actions at 15 Hz before the next inference.

| Task | Success | Failure |
|---|---|---|
| 1 cube | [1](assets/arena/1-cube-success-01-external.mp4) [2](assets/arena/1-cube-success-02-external.mp4) [3](assets/arena/1-cube-success-03-external.mp4) | [1](assets/arena/1-cube-failure-01-external.mp4) [2](assets/arena/1-cube-failure-02-external.mp4) [3](assets/arena/1-cube-failure-03-external.mp4) |
| 2 cubes | [1](assets/arena/2-cubes-success-01-external.mp4) [2](assets/arena/2-cubes-success-02-external.mp4) [3](assets/arena/2-cubes-success-03-external.mp4) | [1](assets/arena/2-cubes-failure-01-external.mp4) [2](assets/arena/2-cubes-failure-02-external.mp4) [3](assets/arena/2-cubes-failure-03-external.mp4) |

Serve the packaged result:

```bash
python3 -m http.server 8000 \
  --directory /workspace/IsaacLab-Scripts/franka_groot_e2e/assets/arena
```
