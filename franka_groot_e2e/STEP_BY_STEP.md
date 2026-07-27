# Franka GR00T end-to-end: step-by-step

This runbook reproduces the validated maximum-two-blue-cube experiment one
stage at a time. For an unattended run, use `run_pipeline.sh` from the main
[README](README.md). Do not mix output directories from different experiments.

## 0. Docker host requirements

The customer environment is a standard Linux Docker host; Brev is not required.
Use the pinned base image `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` so that
Isaac Sim 6.0.1 and Isaac Lab 3.0.0 match the validated software baseline.

Host requirements:

- x86_64 Linux with a compatible NVIDIA driver;
- Docker Engine and NVIDIA Container Toolkit;
- one or more visible CUDA GPUs;
- W&B account for online SFT logging;
- persistent local storage mounted at `/workspace` inside the container. The
  complete validated run used more than 350 GB; 1 TB is recommended.

The maintained source branches are:

| Component | Repository | Branch |
|---|---|---|
| Workflow scripts | `jihyeonRyu/IsaacLab-Scripts` | `main` |
| GR00T SFT | `jihyeonRyu/Isaac-GR00T` | `jryu/franka-demo` |
| Arena evaluation | `jihyeonRyu/IsaacLab-Arena` | `jryu/franka-demo` |

## 1. Start the pinned Docker container and install

On the customer Docker host, create a persistent workspace and start a detached
container. The detached container keeps running if the SSH or terminal session
is closed.

```bash
export HOST_WORKSPACE="$PWD/franka-workspace"
mkdir -p "${HOST_WORKSPACE}"

docker pull nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
docker run -d --name franka-groot-e2e --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "${HOST_WORKSPACE}:/workspace" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  bash -lc "sleep infinity"

docker exec -it franka-groot-e2e bash
```

If `docker pull` requests NGC credentials, log in to `nvcr.io` with user
`$oauthtoken` and an NGC API key. GR00T and Cosmos model downloads themselves
are public and do not require a Hugging Face token.

Run the following commands inside the container:

```bash
nvidia-smi
test -x /isaac-sim/python.sh
test -x /workspace/isaaclab/isaaclab.sh
cat /isaac-sim/VERSION
cat /workspace/isaaclab/VERSION

git clone https://github.com/jihyeonRyu/IsaacLab-Scripts.git \
  /workspace/IsaacLab-Scripts

bash /workspace/IsaacLab-Scripts/franka_groot_e2e/install_franka_groot_e2e.sh \
  --workspace-root /workspace \
  --num-gpus auto \
  --accept-eula
```

The installer refuses the wrong container layout, reuses the bundled Isaac
runtime without reinstalling Isaac Sim/Lab, installs Arena runtime dependencies
in a workspace-local Python user site, creates an isolated GR00T venv, clones
the maintained branches, and downloads both public models. Custom internal
paths can be supplied with `--scripts-repo`, `--groot-repo`, `--arena-repo`, and
`--models-root`.

Load the generated paths and authenticate W&B:

```bash
source /workspace/franka_groot_env.sh
"${GROOT_PYTHON}" -m wandb login
"${GROOT_PYTHON}" -m wandb login --verify
```

Verify the selected GPUs, branches, and model files:

```bash
echo "NUM_GPUS=${NUM_GPUS} GPU_IDS=${GPU_IDS}"
git -C "${SCRIPTS_REPO}" branch --show-current
git -C "${GROOT_REPO}" branch --show-current
git -C "${ARENA_REPO}" branch --show-current
test -f "${BASE_MODEL_PATH}/config.json"
test -f "${GROOT_COSMOS_MODEL_PATH}/config.json"
```

For a subset or non-default physical GPU IDs, rerun the installer with, for
example, `--num-gpus 4 --gpu-ids 0,2,4,6`. All later stages reuse those values.
The reference result used 8 RTX PRO 6000 Blackwell Server Edition GPUs with
approximately 96 GiB VRAM each. On a 44–48 GiB GPU, begin with a per-GPU SFT
batch of 2–4 and increase only after a smoke test.

## 2. Define one experiment

Keep these values in the same shell for all manual stages:

```bash
source /workspace/franka_groot_env.sh

export GENERATION_EPISODES=2000
export GENERATION_SEED=91007
export NUM_ENVS_PER_GPU=4
export PER_GPU_BATCH_SIZE=16
export GLOBAL_BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH_SIZE))
export MAX_STEPS=25000
export EXPERIMENT_NAME=franka-blue-cube-max2-robust-2000-ema
export RAW_DATASET="${WORKSPACE_ROOT}/output/franka_max2_robust_${GENERATION_EPISODES}eps_seed${GENERATION_SEED}"
export LEROBOT_DATASET="${WORKSPACE_ROOT}/datasets/franka_max2_robust_seed${GENERATION_SEED}_lerobot"
export STATE_DIR="${WORKSPACE_ROOT}/output/franka_e2e_pipeline_final"
export RUN_DIR="${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}"
export CHECKPOINT="${RUN_DIR}/checkpoint-${MAX_STEPS}-ema"
export ATTENTION_DIR="${GROOT_REPO}/outputs/attention/${EXPERIMENT_NAME}"
export EVAL_OUTPUT="${ARENA_REPO}/outputs/franka-gr00t-parallel/max2-robust-${GENERATION_EPISODES}-ema-default-start-${NUM_GPUS}gpu-100eps"
IFS=, read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

mkdir -p "${STATE_DIR}"
```

Before a full run, inspect every resolved path without changing data:

```bash
bash "${SCRIPTS_REPO}/franka_groot_e2e/run_pipeline.sh" \
  --workspace-root "${WORKSPACE_ROOT}" \
  --num-gpus "${NUM_GPUS}" \
  --gpu-ids "${GPU_IDS}" \
  --num-envs-per-gpu "${NUM_ENVS_PER_GPU}" \
  --per-gpu-batch-size "${PER_GPU_BATCH_SIZE}" \
  --print-config
```

## 3. Generate synthetic trajectories

This is the validated data distribution: one/two cubes at 25/75%, 30% partial
two-cube continuations, stratified target and start poses, 10% short pre-grasp
near-cube recovery, one solver recovery retry, and no post-grasp wandering.

The output path must be new or empty.

```bash
"${ISAAC_PYTHON}" \
  "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/franka_lift_auto_parallel.py" \
  --headless \
  --enable_cameras \
  --num_envs "${NUM_ENVS_PER_GPU}" \
  --auto_generate_episodes "${GENERATION_EPISODES}" \
  --gpu_ids "${GPU_ID_ARRAY[@]}" \
  --asset_version_override 5.1 \
  --sensor_modalities rgb \
  --output_dir "${RAW_DATASET}" \
  --fps 15 \
  --width 320 \
  --height 256 \
  --no_realtime \
  --seed "${GENERATION_SEED}" \
  --max_blue_cubes 2 \
  --blue_cube_count_weights 0.25 0.75 \
  --workspace_x_min 0.33 \
  --workspace_x_max 0.70 \
  --workspace_y_min -0.34 \
  --workspace_y_max 0.34 \
  --workspace_radius_max 0.68 \
  --stratified_target_positions \
  --target_workspace_bins 4 6 \
  --randomize_start_pose \
  --stratified_start_positions \
  --start_workspace_bins 4 6 3 \
  --start_ee_x_range 0.36 0.70 \
  --start_ee_y_range -0.34 0.34 \
  --start_ee_z_range 0.25 0.55 \
  --start_ee_radius_min 0.40 \
  --start_ee_radius_max 0.72 \
  --start_pose_yaw_tolerance_deg 2.5 \
  --recovery_waypoint_prob 0.10 \
  --recovery_waypoint_radius_range 0.04 0.08 \
  --recovery_waypoint_height_range 0.12 0.18 \
  --partial_progress_2_cube_prob 0.30 \
  --partial_progress_start_xy_radius_range 0.0 0.05 \
  --partial_progress_start_clearance_range 0.12 0.20 \
  --partial_progress_start_yaw_range_deg -45 45 \
  --solver_recovery_max_attempts 1
```

Validate every selected worker before continuing:

```bash
"${ISAAC_PYTHON}" - "${RAW_DATASET}/multi_gpu_summary.json" "${GENERATION_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
assert summary["requested_episodes"] == expected, summary
assert summary["reported_episodes"] == expected, summary
assert summary["all_workers_exited_cleanly"] is True, summary
print(json.dumps(summary, indent=2))
PY
```

`attempts` and successful training episodes are different: failed solver or
transport attempts remain available for analysis, while only successful
trajectories are converted to LeRobot.

## 4. Analyze generation

```bash
"${ISAAC_PYTHON}" \
  "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/analyze_franka_trajectories.py" \
  "${RAW_DATASET}" \
  --output-dir "${RAW_DATASET}/trajectory_analysis" \
  --max-blue-cubes 2 \
  --strict
```

Inspect the machine-readable scenario and failure summaries:

```bash
"${ISAAC_PYTHON}" -m json.tool \
  "${RAW_DATASET}/trajectory_analysis/scenario_summary.json"
ls -lh "${RAW_DATASET}/trajectory_analysis/"*.png
```

The validated run produced 1,757 successful trajectories from 2,000 attempts:
478/495 for one cube and 1,279/1,505 for two cubes.

## 5. Convert successful episodes to LeRobot

The output path must be new or empty.

```bash
"${GROOT_PYTHON}" \
  "${SCRIPTS_REPO}/franka_groot_e2e/scripts/02_convert/convert_franka_to_groot_lerobot.py" \
  "${RAW_DATASET}" \
  "${LEROBOT_DATASET}"
```

Validate the metadata:

```bash
test -f "${LEROBOT_DATASET}/meta/info.json"
test -f "${LEROBOT_DATASET}/meta/episodes.jsonl"
"${GROOT_PYTHON}" -m json.tool "${LEROBOT_DATASET}/meta/info.json"
```

The validated dataset contains 1,757 episodes, 754,545 frames at 15 FPS, and
686,022 valid 40-frame training windows.

## 6. Audit frame coverage and choose SFT steps

The audit prevents choosing steps from episode count alone. The validated
8-GPU/global-batch-128 run needs 5,360 optimizer steps for one nominal pass.
When GPU count or per-GPU batch changes, rerun this audit and target at least
4.5 nominal passes; `run_pipeline.sh --max-steps auto` performs that calculation.

```bash
"${GROOT_PYTHON}" \
  "${GROOT_REPO}/tools/audit_franka_training_coverage.py" \
  --episodes "${LEROBOT_DATASET}/meta/episodes.jsonl" \
  --action-horizon 40 \
  --shard-size 512 \
  --episode-sampling-rate 0.1 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --output "${STATE_DIR}/frame_coverage_audit.json" \
  --format json

"${GROOT_PYTHON}" -m json.tool "${STATE_DIR}/frame_coverage_audit.json"
```

For the published global batch 128 run, 25,000 steps correspond to about 4.66
nominal data passes. Keep data passes comparable when changing batch size.

## 7. Train GR00T on the selected GPUs

The four debug probes below are the validated continuation-aware samples. For a
custom dataset, choose four episodes longer than frame 160 and keep the same
episode/frame IDs for the first-checkpoint and final-EMA comparison.
`RUN_DIR` and `ATTENTION_DIR` must be new or empty for a new experiment.

```bash
export DEBUG_VIS_EPISODES="1 0 3 2"

DATASET_PATH="${LEROBOT_DATASET}" \
VENV_PATH="${GROOT_REPO}/.venv" \
BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
OUTPUT_DIR="${GROOT_REPO}/outputs/franka-groot-sft" \
HF_HOME="${HF_HOME}" \
GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
NUM_GPUS="${NUM_GPUS}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=2 \
LEARNING_RATE=1e-4 \
LR_SCHEDULER_TYPE=cosine \
WARMUP_RATIO=0.05 \
WEIGHT_DECAY=1e-5 \
USE_EMA=1 \
EMA_DECAY=0.999 \
EMA_UPDATE_AFTER_STEP=0 \
EMA_UPDATE_EVERY=1 \
SHARD_SIZE=512 \
EPISODE_SAMPLING_RATE=0.1 \
NUM_SHARDS_PER_EPOCH=1340 \
SHORTEST_IMAGE_EDGE=256 \
CROP_FRACTION=0.98 \
STATE_DROPOUT_PROB=0.2 \
PROCESSOR_STATE_DROPOUT_PROB=0.0 \
COLOR_JITTER_PARAMS="brightness 0.25 contrast 0.25 saturation 0.30 hue 0.03" \
WANDB_MODE=online \
DEBUG_VISUALIZE=1 \
DEBUG_VIS_EPISODES="${DEBUG_VIS_EPISODES}" \
DEBUG_VIS_FRAME_STEP=120 \
DEBUG_VIS_EVERY_N_CHECKPOINTS=1 \
bash "${GROOT_REPO}/examples/Franka/train_franka.sh"
```

Validate the final EMA checkpoint and the four first saved checkpoint probes:

```bash
test -f "${CHECKPOINT}/config.json"
test -f "${CHECKPOINT}/ema_config.json"
find "${ATTENTION_DIR}" -maxdepth 1 \
  -name 'checkpoint-1000-ep*-step120.png' -print
```

The expected first-probe count is four. Training should not proceed to Arena if
the final EMA checkpoint is incomplete.

## 8. Render matched final-EMA attention

Render the same four episode/frame samples at the final EMA checkpoint:

```bash
for episode in ${DEBUG_VIS_EPISODES}; do
  CUDA_VISIBLE_DEVICES="${GPU_ID_ARRAY[0]}" WANDB_MODE=offline "${GROOT_PYTHON}" \
    "${GROOT_REPO}/tools/visualize_franka_attention.py" \
    --dataset "${LEROBOT_DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --episode "${episode}" \
    --step 120 \
    --action-group all \
    --device cuda:0 \
    --output "${ATTENTION_DIR}/final-ema-episode-${episode}-step-120.png" \
    --wandb-project franka-groot-sft \
    --wandb-run-name "${EXPERIMENT_NAME}-ema-debug-ep${episode}" \
    --global-step "${MAX_STEPS}" \
    --full-reasoner-model "${GROOT_COSMOS_MODEL_PATH}"
done
```

Verify four PNG and four JSON files for each side of the comparison:

```bash
find "${ATTENTION_DIR}" -maxdepth 1 -name 'checkpoint-1000-ep*-step120.png' | wc -l
find "${ATTENTION_DIR}" -maxdepth 1 -name 'final-ema-episode-*-step-120.png' | wc -l
```

## 9. Evaluate in IsaacLab-Arena

Arena starts one policy server and one simulator worker per GPU. Evaluation uses
the generation-matched task geometry, cameras, objects, tray, and lighting, but
uses independent seeds. The Franka starts from the fixed default pose. GR00T
predicts 40 actions; Arena executes 16 at 15 Hz before the next inference.

The output path must be new or empty.

```bash
"${GROOT_PYTHON}" \
  "${ARENA_REPO}/isaaclab_arena_gr00t/parallel_evaluation.py" \
  --checkpoint "${CHECKPOINT}" \
  --num-gpus "${NUM_GPUS}" \
  --gpu-ids "${GPU_IDS}" \
  --episodes-per-task 100 \
  --task franka_blue_tray_1_cube \
  --task franka_blue_tray_2_cubes \
  --base-port 5955 \
  --arena-repo "${ARENA_REPO}" \
  --gr00t-repo "${GROOT_REPO}" \
  --cosmos-model-path "${GROOT_COSMOS_MODEL_PATH}" \
  --arena-python "${ARENA_PYTHON}" \
  --gr00t-python "${GROOT_PYTHON}" \
  --no-randomize-policy-start-pose \
  --output-dir "${EVAL_OUTPUT}"
```

Validate that each task has exactly 100 completed episodes:

```bash
"${GROOT_PYTHON}" - "${EVAL_OUTPUT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
expected = {"franka_blue_tray_1_cube", "franka_blue_tray_2_cubes"}
assert set(summary) == expected, summary
for task, result in summary.items():
    assert result["episodes"] == 100, (task, result)
print(json.dumps(summary, indent=2))
PY
```

The validated fixed-default-pose result is 98/100 for one cube and 70/100 for
two cubes. Arena uses the generator-matched center-within-tray 15 mm margin.

## 10. Package the evidence and customer docs

This copies plots, two generation examples, the four matched first/final
attention pairs, and three success plus three failure videos per Arena task into
`franka_groot_e2e/assets`. It also creates compact full-episode GIF previews so
all media renders inline in GitHub, then regenerates the result README. It does
not commit, push, upload, or contact an external service.

```bash
python3 \
  "${SCRIPTS_REPO}/franka_groot_e2e/scripts/05_finalize/finalize_franka_run.py" \
  --workspace-root "${WORKSPACE_ROOT}" \
  --scripts-repo "${SCRIPTS_REPO}" \
  --raw-dataset "${RAW_DATASET}" \
  --lerobot-dataset "${LEROBOT_DATASET}" \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --attention-dir "${ATTENTION_DIR}" \
  --arena-output "${EVAL_OUTPUT}" \
  --state-dir "${STATE_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --generation-attempts "${GENERATION_EPISODES}" \
  --generation-seed "${GENERATION_SEED}" \
  --num-gpus "${NUM_GPUS}" \
  --num-envs-per-gpu "${NUM_ENVS_PER_GPU}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
```

Audit the packaged counts:

```bash
find "${SCRIPTS_REPO}/franka_groot_e2e/assets/attention" -maxdepth 1 -name '*.png' | wc -l
find "${SCRIPTS_REPO}/franka_groot_e2e/assets/arena" -maxdepth 1 -name '*.mp4' | wc -l
find "${SCRIPTS_REPO}/franka_groot_e2e/assets" -name '*.gif' | wc -l
```

The expected counts are eight attention PNGs, twelve Arena MP4s, and fourteen
animated GIF previews (twelve Arena plus two generation).

## 11. Open the evaluation result

```bash
python3 -m http.server 8000 \
  --directory "${SCRIPTS_REPO}/franka_groot_e2e/assets/arena"
```

Open `http://SERVER_IP:8000/`. Stop the server with `Ctrl+C`.

## Restart and failure rules

- The one-command pipeline records stage markers and is restart-safe when given
  the same `STATE_DIR` and output paths.
- Manual stage commands should write to new output directories after a failed
  partial run. Do not merge partial raw, LeRobot, SFT, or Arena outputs.
- Keep the generation seed separate from Arena seeds.
- Do not enable randomized Arena policy starts for the validated benchmark.
- Do not evaluate a non-EMA or incomplete checkpoint.
- Check `multi_gpu_summary.json`, `frame_coverage_audit.json`,
  `ema_config.json`, and Arena `summary.json` before advancing.
