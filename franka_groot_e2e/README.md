# Franka synthetic data → GR00T → IsaacLab-Arena

This directory is the single reproducible workflow for the Franka blue-cube
pick-and-place task. It installs the three repositories and models, generates a
new synthetic dataset, analyzes it, converts successful trajectories to LeRobot
v2.1, fine-tunes GR00T N1.7 with EMA on eight GPUs, and evaluates the policy in
IsaacLab-Arena.

Only the current production path is documented here. Historical experiment launchers and their stale outputs are intentionally not retained.

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
    └── README.md
```

The launcher uses these source branches:

- `jihyeonRyu/Isaac-GR00T`, branch `jryu/franka-demo`
- `jihyeonRyu/IsaacLab-Arena`, branch `jryu/franka-demo`
- `jihyeonRyu/IsaacLab-Scripts`, branch `main`

## 1. Recommended environment

Use the same NVIDIA/Isaac Lab Docker image used for generation and Arena. The
host must expose eight CUDA GPUs and enough writable disk for roughly 200 GB of
raw videos plus model checkpoints. The installer creates isolated environments
rather than modifying the system Python:

- `/workspace/env_isaaclab`: Isaac Lab generation and Arena workers
- `/workspace/Isaac-GR00T/.venv`: conversion, SFT, inference server, analysis
- `/workspace/IsaacLab-Arena/.venv`: Arena coordinator
- `/workspace/.tools/ffmpeg-7`: TorchCodec-compatible FFmpeg runtime

From an existing `IsaacLab-Scripts` checkout:

```bash
bash /workspace/IsaacLab-Scripts/franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /workspace \
  --accept-eula
```

Every path can be overridden. Inspect the resolved installation without making
changes:

```bash
bash franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /customer/workspace \
  --scripts-repo /customer/workspace/IsaacLab-Scripts \
  --groot-repo /customer/workspace/Isaac-GR00T \
  --arena-repo /customer/workspace/IsaacLab-Arena \
  --models-root /customer/workspace/models \
  --print-config
```

The installer checks out the required branches, installs each repository in its
own virtual environment, installs the RPC/video dependencies, and downloads the
public `GR00T-N1.7-3B` and `Cosmos-Reason2-2B` weights. Hugging Face login is not
required for these models. W&B authentication is separate:

```bash
/workspace/Isaac-GR00T/.venv/bin/wandb login
```

## 2. Run the final pipeline

The default production run is detached so it continues after the terminal or
client computer disconnects:

```bash
cd /workspace/IsaacLab-Scripts
nohup bash franka_groot_e2e/run_pipeline.sh \
  --workspace-root /workspace \
  > /workspace/output/franka_final_pipeline.log 2>&1 &
echo $! > /workspace/output/franka_final_pipeline.pid
```

Inspect paths and settings without starting work:

```bash
bash franka_groot_e2e/run_pipeline.sh --print-config
```

Monitor the supervisor:

```bash
tail -f /workspace/output/franka_final_pipeline.log
cat /workspace/output/franka_e2e_pipeline_final/status.log
```

Completion is recorded at:

```text
/workspace/output/franka_e2e_pipeline_final/complete.done
```

The pipeline is restart-safe at completed stage boundaries. It refuses to
silently overwrite a non-empty partial raw dataset, LeRobot dataset, training
run, or Arena result.

## 3. Final generation specification

The launcher records 1,200 attempts with seed `90007` using 8 GPUs and four
vector environments per GPU. Evaluation uses independent seeds.

Core settings:

- 15 FPS, 320×256 RGB from `external` and `wrist` cameras;
- fixed validated camera geometry;
- blue targets stratified across a 4×6 X/Y workspace grid;
- workspace X `0.33–0.70 m`, Y `-0.34–0.34 m`, radial limit `0.68 m`;
- translated start EEF X `0.36–0.70 m`, Y `-0.34–0.34 m`, Z `0.25–0.55 m`;
- the validated floor-facing tool orientation is preserved;
- cube/tray/light/background randomization is enabled;
- background RGB is sampled over the full `[0,1]³` range;
- post-yaw XY recentering is enforced before descent;
- successful pre-grasp recovery waypoint probability is 10%, radius 4–8 cm;
- failure-triggered solver recovery is limited to one retry per cube;
- only successful episodes are converted for SFT.

### Partial-progress continuation curriculum

This directly covers the visual state that occurs before a second or third
pickup:

- one cube: 100% full start;
- two cubes: 75% full start, 25% one cube already in the tray;
- three cubes: 70% full start, 18% one preplaced, 12% two preplaced.

For partial starts, a random subset occupies the first completed tray slots and
the EEF starts in a post-placement retreat region 0–5 cm from the last completed
slot and 12–20 cm above that cube. The controller begins at the next free tray
slot and manipulates only remaining loose targets. Loose objects must still be
outside the tray footprint; preplaced targets are validated inside the tray
walls at the correct support height.

Every raw `scenario.json` records:

```text
progress_stage
num_blue_total
num_preplaced
num_remaining
preplaced_blue_cube_names
remaining_blue_cube_names
start_pose_mode
```

## 4. Analysis and LeRobot conversion

After generation, the analyzer writes:

```text
trajectory_analysis/
├── scenario_summary.json
├── episode_metrics.csv
├── scenario_success.csv
├── progress_stage_success.csv
├── failure_causes.csv
├── solver_recovery_outcomes.csv
├── workspace_coverage.csv
├── trajectory_distribution.png
├── trajectory_by_blue_cube_count.png
├── workspace_coverage.png
├── scenario_statistics.png
├── progress_stage_statistics.png
└── failure_analysis.png
```

Preplaced tray coordinates are excluded from loose-target workspace coverage.
The pipeline fails before conversion unless all six required groups exist:
`1c/0p`, `2c/0p`, `2c/1p`, `3c/0p`, `3c/1p`, and `3c/2p`.

The converter retains progress metadata in `meta/episodes.jsonl`, which makes
continuation samples auditable and allows attention probes to select the
intended states rather than arbitrary episode indices.

LeRobot data contract:

- `observation.state`: absolute EEF XYZ + rotation 6D + gripper width (10D);
- `action`: stored delta XYZ + delta rotvec + absolute gripper command (7D);
- task: `Pick up every blue cube and place it in the green tray. Ignore the red cubes.`;
- training action horizon: 40 frames, about 2.67 s at 15 FPS;
- percentile normalization is enabled.

## 5. GR00T SFT

The final trainer uses:

- GR00T N1.7 3B and Cosmos Reason2 2B;
- 8 GPUs, global batch 128 (16 per GPU);
- automatic exhaustive balanced sharding;
- automatic step count targeting 4.5 nominal passes over every valid
  `action[t:t+40]` window;
- LR `1e-4`, cosine decay, 5% warmup, weight decay `1e-5`;
- crop fraction `0.98`;
- brightness `0.25`, contrast `0.25`, saturation `0.30`, hue `0.03`;
- action-head state dropout `0.2`, processor state dropout `0.0`;
- FP32 EMA decay `0.999`, updated every optimizer step;
- online W&B logging;
- checkpoint interval 1,000 steps, at most two retained during training.

Four debug episodes are chosen from partial-progress metadata, including
2-cube/1-preplaced and 3-cube continuation cases. The final attention artifacts
are rendered from the EMA checkpoint at frame 120. After the full pipeline
passes, non-final raw checkpoints and intermediate attention maps are removed.

## 6. IsaacLab-Arena evaluation

Arena launches one GR00T inference server and one simulation worker per GPU.
It runs 100 full-task episodes for each of 1, 2, and 3 blue cubes. Camera,
object, tray, lighting, action adapter, and start-pose distributions match
training generation; evaluation seeds differ from seed `90007`.

GR00T predicts 40 actions. Arena executes the first 16 at the dataset-matched
15 Hz before requesting a new chunk. The final result is accepted only when all
three tasks contain exactly 100 episode records and `summary.json` plus
`index.html` are present.

## 7. Final artifacts

`assets/` is intentionally empty of historical experiment evidence. After the
active 1,200-attempt pipeline completes and passes validation, it is populated
only with:

- two representative successful generation videos;
- final trajectory/workspace/progress/failure analysis figures;
- four continuation-aware final EMA attention maps;
- the complete Arena summary and HTML report;
- one success and one failure video per cube count when both outcomes exist.

The result table and measured dataset statistics are added here only after the
final outputs pass their completeness checks.
