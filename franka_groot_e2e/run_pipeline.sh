#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCRIPTS_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
SCRIPTS_REPO="${SCRIPTS_REPO:-}"
GROOT_REPO="${GROOT_REPO:-}"
ARENA_REPO="${ARENA_REPO:-}"
MODELS_ROOT="${MODELS_ROOT:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH:-}"
HF_HOME="${HF_HOME:-}"
FFMPEG_RUNTIME="${FFMPEG_RUNTIME:-}"
ISAAC_PYTHON="${ISAAC_PYTHON:-}"
ARENA_PYTHON="${ARENA_PYTHON:-}"
GROOT_PYTHON="${GROOT_PYTHON:-}"
RAW_DATASET="${RAW_DATASET:-}"
LEROBOT_DATASET="${LEROBOT_DATASET:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-partial-progress-1200-ema}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}"
STATE_DIR="${STATE_DIR:-}"
GENERATION_EPISODES="${GENERATION_EPISODES:-1200}"
GENERATION_SEED="${GENERATION_SEED:-90007}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-auto}"
TARGET_DATA_PASSES="${TARGET_DATA_PASSES:-4.5}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
SHARD_SIZE="${SHARD_SIZE:-512}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
EMA_DECAY="${EMA_DECAY:-0.999}"
CROP_FRACTION="${CROP_FRACTION:-0.98}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"
PROCESSOR_STATE_DROPOUT_PROB="${PROCESSOR_STATE_DROPOUT_PROB:-0.0}"
COLOR_JITTER_PARAMS="${COLOR_JITTER_PARAMS:-brightness 0.25 contrast 0.25 saturation 0.30 hue 0.03}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
PRINT_CONFIG=0

usage() {
    cat <<'EOF'
Run the final Franka pipeline:
  8-GPU generation -> analysis -> LeRobot -> 8-GPU GR00T SFT+EMA -> Arena.

Usage:
  run_pipeline.sh [options]

Paths:
  --workspace-root PATH
  --scripts-repo PATH
  --groot-repo PATH
  --arena-repo PATH
  --models-root PATH
  --base-model-path PATH
  --cosmos-model-path PATH
  --hf-home PATH
  --ffmpeg-runtime PATH
  --isaac-python PATH
  --arena-python PATH
  --groot-python PATH
  --raw-dataset PATH
  --lerobot-dataset PATH
  --eval-output PATH
  --state-dir PATH

Run settings:
  --generation-episodes N     Default: 1200 attempts
  --generation-seed N         Default: 90007
  --global-batch-size N       Default: 128 across 8 GPUs
  --max-steps auto|N          Default: auto
  --target-data-passes VALUE  Default: 4.5 when max-steps=auto
  --experiment-name NAME
  --wait-for-pid PID

Other:
  --groot-branch NAME
  --arena-branch NAME
  --print-config
  -h, --help

Fixed task settings:
  2-cube partial progress 25%; 3-cube partial progress 30%, with 40% of
  partial 3-cube episodes starting with two cubes preplaced. Near-cube recovery
  waypoint probability is 10%, solver recovery is limited to one retry, model
  horizon is 40, and Arena executes 16 actions at 15 Hz.
EOF
}

need_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

normalize_executable_path() {
    local executable_path=$1
    local executable_dir
    executable_dir="$(realpath -m -- "$(dirname -- "${executable_path}")")"
    printf '%s/%s\n' "${executable_dir}" "$(basename -- "${executable_path}")"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workspace-root) need_value "$@"; WORKSPACE_ROOT=$2; shift 2 ;;
        --scripts-repo) need_value "$@"; SCRIPTS_REPO=$2; shift 2 ;;
        --groot-repo) need_value "$@"; GROOT_REPO=$2; shift 2 ;;
        --arena-repo) need_value "$@"; ARENA_REPO=$2; shift 2 ;;
        --models-root) need_value "$@"; MODELS_ROOT=$2; shift 2 ;;
        --base-model-path) need_value "$@"; BASE_MODEL_PATH=$2; shift 2 ;;
        --cosmos-model-path) need_value "$@"; GROOT_COSMOS_MODEL_PATH=$2; shift 2 ;;
        --hf-home) need_value "$@"; HF_HOME=$2; shift 2 ;;
        --ffmpeg-runtime) need_value "$@"; FFMPEG_RUNTIME=$2; shift 2 ;;
        --isaac-python) need_value "$@"; ISAAC_PYTHON=$2; shift 2 ;;
        --arena-python) need_value "$@"; ARENA_PYTHON=$2; shift 2 ;;
        --groot-python) need_value "$@"; GROOT_PYTHON=$2; shift 2 ;;
        --raw-dataset) need_value "$@"; RAW_DATASET=$2; shift 2 ;;
        --lerobot-dataset) need_value "$@"; LEROBOT_DATASET=$2; shift 2 ;;
        --eval-output) need_value "$@"; EVAL_OUTPUT=$2; shift 2 ;;
        --state-dir) need_value "$@"; STATE_DIR=$2; shift 2 ;;
        --generation-episodes) need_value "$@"; GENERATION_EPISODES=$2; shift 2 ;;
        --generation-seed) need_value "$@"; GENERATION_SEED=$2; shift 2 ;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE=$2; shift 2 ;;
        --max-steps) need_value "$@"; MAX_STEPS=$2; shift 2 ;;
        --target-data-passes) need_value "$@"; TARGET_DATA_PASSES=$2; shift 2 ;;
        --experiment-name) need_value "$@"; EXPERIMENT_NAME=$2; shift 2 ;;
        --wait-for-pid) need_value "$@"; WAIT_FOR_PID=$2; shift 2 ;;
        --groot-branch) need_value "$@"; GROOT_BRANCH=$2; shift 2 ;;
        --arena-branch) need_value "$@"; ARENA_BRANCH=$2; shift 2 ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in GENERATION_EPISODES GLOBAL_BATCH_SIZE SAVE_STEPS SAVE_TOTAL_LIMIT ACTION_HORIZON; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done
if ! [[ "${GENERATION_SEED}" =~ ^[0-9]+$ ]]; then
    echo "GENERATION_SEED must be a non-negative integer" >&2
    exit 2
fi
if [ "${MAX_STEPS}" != "auto" ] && ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be auto or a positive integer" >&2
    exit 2
fi
if [ "$((GLOBAL_BATCH_SIZE % 8))" -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be divisible by 8" >&2
    exit 2
fi
if [ -n "${WAIT_FOR_PID}" ] && ! [[ "${WAIT_FOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WAIT_FOR_PID must be a positive integer" >&2
    exit 2
fi
python3 - "${TARGET_DATA_PASSES}" <<'PY'
import sys
value = float(sys.argv[1])
if not value >= 1.0:
    raise SystemExit("TARGET_DATA_PASSES must be >= 1.0")
PY

WORKSPACE_ROOT="$(realpath -m -- "${WORKSPACE_ROOT}")"
SCRIPTS_REPO="$(realpath -m -- "${SCRIPTS_REPO:-${DEFAULT_SCRIPTS_REPO}}")"
GROOT_REPO="$(realpath -m -- "${GROOT_REPO:-${WORKSPACE_ROOT}/Isaac-GR00T}")"
ARENA_REPO="$(realpath -m -- "${ARENA_REPO:-${WORKSPACE_ROOT}/IsaacLab-Arena}")"
MODELS_ROOT="$(realpath -m -- "${MODELS_ROOT:-${WORKSPACE_ROOT}/models}")"
BASE_MODEL_PATH="$(realpath -m -- "${BASE_MODEL_PATH:-${MODELS_ROOT}/GR00T-N1.7-3B}")"
GROOT_COSMOS_MODEL_PATH="$(realpath -m -- "${GROOT_COSMOS_MODEL_PATH:-${MODELS_ROOT}/Cosmos-Reason2-2B}")"
HF_HOME="$(realpath -m -- "${HF_HOME:-${MODELS_ROOT}/huggingface-cache}")"
FFMPEG_RUNTIME="$(realpath -m -- "${FFMPEG_RUNTIME:-${WORKSPACE_ROOT}/.tools/ffmpeg-7}")"
ISAAC_PYTHON="$(normalize_executable_path "${ISAAC_PYTHON:-${WORKSPACE_ROOT}/env_isaaclab/bin/python}")"
ARENA_PYTHON="$(normalize_executable_path "${ARENA_PYTHON:-${ISAAC_PYTHON}}")"
GROOT_PYTHON="$(normalize_executable_path "${GROOT_PYTHON:-${GROOT_REPO}/.venv/bin/python}")"
RAW_DATASET="$(realpath -m -- "${RAW_DATASET:-${WORKSPACE_ROOT}/output/franka_partial_progress_${GENERATION_EPISODES}eps_seed${GENERATION_SEED}}")"
LEROBOT_DATASET="$(realpath -m -- "${LEROBOT_DATASET:-${WORKSPACE_ROOT}/datasets/franka_partial_progress_seed${GENERATION_SEED}_lerobot}")"
EVAL_OUTPUT="$(realpath -m -- "${EVAL_OUTPUT:-${ARENA_REPO}/outputs/franka-gr00t-parallel/partial-progress-1200-ema-default-start-8gpu-100eps}")"
STATE_DIR="$(realpath -m -- "${STATE_DIR:-${WORKSPACE_ROOT}/output/franka_e2e_pipeline_final}")"
RUN_DIR="${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}"
ATTENTION_DIR="${GROOT_REPO}/outputs/attention/${EXPERIMENT_NAME}"

print_config() {
    cat <<EOF
Resolved final Franka pipeline
  workspace          : ${WORKSPACE_ROOT}
  scripts/GR00T/Arena: ${SCRIPTS_REPO} / ${GROOT_REPO} / ${ARENA_REPO}
  raw dataset        : ${RAW_DATASET}
  LeRobot dataset    : ${LEROBOT_DATASET}
  generation         : ${GENERATION_EPISODES} attempts, seed ${GENERATION_SEED}, 8 GPUs x 4 envs
  partial progress   : 2c=25%; 3c=30% (conditional two-preplaced=40%)
  experiment         : ${EXPERIMENT_NAME}
  batch              : ${GLOBAL_BATCH_SIZE} global ($((GLOBAL_BATCH_SIZE / 8)) per GPU)
  steps              : ${MAX_STEPS} (target passes=${TARGET_DATA_PASSES})
  checkpoints        : every ${SAVE_STEPS}; retain ${SAVE_TOTAL_LIMIT}; final EMA only after completion
  augmentation       : crop=${CROP_FRACTION}; jitter=${COLOR_JITTER_PARAMS}; state dropout=${STATE_DROPOUT_PROB}
  action horizon     : train=${ACTION_HORIZON}; Arena execute=16
  Arena output       : ${EVAL_OUTPUT}
  state/log directory: ${STATE_DIR}
EOF
}

print_config
if [ "${PRINT_CONFIG}" = "1" ]; then
    exit 0
fi

for required_path in \
    "${ISAAC_PYTHON}" \
    "${ARENA_PYTHON}" \
    "${GROOT_PYTHON}" \
    "${BASE_MODEL_PATH}/config.json" \
    "${GROOT_COSMOS_MODEL_PATH}/config.json" \
    "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/franka_lift_auto_parallel.py" \
    "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/analyze_franka_trajectories.py" \
    "${SCRIPTS_REPO}/franka_groot_e2e/scripts/02_convert/convert_franka_to_groot_lerobot.py" \
    "${GROOT_REPO}/examples/Franka/train_franka.sh" \
    "${GROOT_REPO}/tools/audit_franka_training_coverage.py" \
    "${GROOT_REPO}/tools/visualize_franka_attention.py" \
    "${ARENA_REPO}/isaaclab_arena_gr00t/parallel_evaluation.py"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done
if [ "$(git -C "${GROOT_REPO}" branch --show-current)" != "${GROOT_BRANCH}" ]; then
    echo "GR00T repo must be on ${GROOT_BRANCH}" >&2
    exit 2
fi
if [ "$(git -C "${ARENA_REPO}" branch --show-current)" != "${ARENA_BRANCH}" ]; then
    echo "Arena repo must be on ${ARENA_BRANCH}" >&2
    exit 2
fi
if [ ! -x "${FFMPEG_RUNTIME}/bin/ffmpeg" ]; then
    echo "FFmpeg runtime is missing: ${FFMPEG_RUNTIME}/bin/ffmpeg" >&2
    exit 2
fi
if ! grep -q -- 'SAVE_TOTAL_LIMIT' "${GROOT_REPO}/examples/finetune.sh"; then
    echo "GR00T checkout lacks configurable checkpoint retention" >&2
    exit 2
fi
if ! "${GROOT_REPO}/.venv/bin/wandb" login --verify >/dev/null 2>&1; then
    echo "W&B login verification failed. Run: ${GROOT_REPO}/.venv/bin/wandb login" >&2
    exit 3
fi

export PATH="${FFMPEG_RUNTIME}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${FFMPEG_RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LOCAL_ISAAC_LIB="${WORKSPACE_ROOT}/.tools/isaac-system-libs/usr/lib/x86_64-linux-gnu"
if [ -d "${LOCAL_ISAAC_LIB}" ]; then
    export LD_LIBRARY_PATH="${LOCAL_ISAAC_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export HF_HOME GROOT_COSMOS_MODEL_PATH
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export PYTHONUNBUFFERED=1
if ! "${GROOT_PYTHON}" -c 'from torchcodec.decoders import VideoDecoder' >/dev/null; then
    echo "TorchCodec cannot load the FFmpeg runtime at ${FFMPEG_RUNTIME}" >&2
    exit 2
fi

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/pipeline.lock"
flock -n 9 || {
    echo "Another Franka pipeline holds ${STATE_DIR}/pipeline.lock" >&2
    exit 1
}

status() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${STATE_DIR}/status.log"
}
mark_done() {
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/$1.done"
}
on_error() {
    local exit_code=$?
    status "FAILED stage=${CURRENT_STAGE:-startup} exit_code=${exit_code} line=${BASH_LINENO[0]}"
    exit "${exit_code}"
}
trap on_error ERR

if [ -n "${WAIT_FOR_PID}" ]; then
    status "waiting for PID ${WAIT_FOR_PID}"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do sleep 30; done
fi

CURRENT_STAGE="generation"
if [ ! -f "${STATE_DIR}/generation.done" ]; then
    if [ ! -f "${RAW_DATASET}/multi_gpu_summary.json" ]; then
        if [ -d "${RAW_DATASET}" ] && [ -n "$(find "${RAW_DATASET}" -mindepth 1 -print -quit)" ]; then
            echo "Refusing incomplete non-empty raw dataset: ${RAW_DATASET}" >&2
            exit 2
        fi
        status "starting 8-GPU generation: ${GENERATION_EPISODES} attempts"
        "${ISAAC_PYTHON}" \
            "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/franka_lift_auto_parallel.py" \
            --headless \
            --enable_cameras \
            --num_envs 4 \
            --auto_generate_episodes "${GENERATION_EPISODES}" \
            --gpu_ids 0 1 2 3 4 5 6 7 \
            --asset_version_override 5.1 \
            --sensor_modalities rgb \
            --output_dir "${RAW_DATASET}" \
            --fps 15 \
            --width 320 \
            --height 256 \
            --no_realtime \
            --seed "${GENERATION_SEED}" \
            --workspace_x_min 0.33 \
            --workspace_x_max 0.70 \
            --workspace_y_min -0.34 \
            --workspace_y_max 0.34 \
            --workspace_radius_max 0.68 \
            --stratified_target_positions \
            --target_workspace_bins 4 6 \
            --randomize_start_pose \
            --start_ee_x_range 0.36 0.70 \
            --start_ee_y_range -0.34 0.34 \
            --start_ee_z_range 0.25 0.55 \
            --start_ee_radius_min 0.40 \
            --start_ee_radius_max 0.72 \
            --recovery_waypoint_prob 0.10 \
            --recovery_waypoint_radius_range 0.04 0.08 \
            --recovery_waypoint_height_range 0.12 0.18 \
            --partial_progress_2_cube_prob 0.25 \
            --partial_progress_3_cube_prob 0.30 \
            --partial_progress_3_cube_two_preplaced_prob 0.40 \
            --partial_progress_start_xy_radius_range 0.0 0.05 \
            --partial_progress_start_clearance_range 0.12 0.20 \
            --solver_recovery_max_attempts 1
    fi
    "${ISAAC_PYTHON}" - "${RAW_DATASET}/multi_gpu_summary.json" "${GENERATION_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
assert summary["requested_episodes"] == expected, summary
assert summary["reported_episodes"] == expected, summary
assert summary["all_workers_exited_cleanly"] is True, summary
print(json.dumps(summary, indent=2))
PY
    mark_done generation
fi

CURRENT_STAGE="analysis"
if [ ! -f "${STATE_DIR}/analysis.done" ]; then
    status "analyzing trajectories, progress stages, workspace, and failures"
    "${ISAAC_PYTHON}" \
        "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/analyze_franka_trajectories.py" \
        "${RAW_DATASET}" \
        --output-dir "${RAW_DATASET}/trajectory_analysis" \
        --strict
    "${ISAAC_PYTHON}" - "${RAW_DATASET}/trajectory_analysis/scenario_summary.json" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
rows = summary["progress_stage_statistics"]
seen = {(row["blue_cube_count"], row["num_preplaced"]) for row in rows}
required = {(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)}
assert required <= seen, (required - seen, rows)
print(json.dumps(rows, indent=2))
PY
    mark_done analysis
fi

CURRENT_STAGE="conversion"
if [ ! -f "${STATE_DIR}/conversion.done" ]; then
    if [ -d "${LEROBOT_DATASET}" ] && [ -n "$(find "${LEROBOT_DATASET}" -mindepth 1 -print -quit)" ]; then
        echo "Refusing non-empty LeRobot output: ${LEROBOT_DATASET}" >&2
        exit 2
    fi
    status "converting successful episodes to LeRobot v2.1"
    "${GROOT_PYTHON}" \
        "${SCRIPTS_REPO}/franka_groot_e2e/scripts/02_convert/convert_franka_to_groot_lerobot.py" \
        "${RAW_DATASET}" \
        "${LEROBOT_DATASET}"
    test -f "${LEROBOT_DATASET}/meta/info.json"
    test -f "${LEROBOT_DATASET}/meta/episodes.jsonl"
    mark_done conversion
fi

CURRENT_STAGE="training_plan"
IFS=$'\t' read -r \
    VALID_TRAINING_WINDOWS \
    NUM_SHARDS_PER_EPOCH \
    MINIMUM_STEPS_FOR_ONE_PASS \
    _COMPLETE_PASSES \
    _NOMINAL_PASSES < <(
        "${GROOT_PYTHON}" \
            "${GROOT_REPO}/tools/audit_franka_training_coverage.py" \
            --episodes "${LEROBOT_DATASET}/meta/episodes.jsonl" \
            --action-horizon "${ACTION_HORIZON}" \
            --shard-size "${SHARD_SIZE}" \
            --episode-sampling-rate "${EPISODE_SAMPLING_RATE}" \
            --global-batch-size "${GLOBAL_BATCH_SIZE}" \
            --max-steps 1 \
            --allow-incomplete-pass \
            --output "${STATE_DIR}/frame_coverage_probe.json" \
            --format tsv
    )
if [ "${MAX_STEPS}" = "auto" ]; then
    MAX_STEPS="$("${GROOT_PYTHON}" - "${MINIMUM_STEPS_FOR_ONE_PASS}" "${TARGET_DATA_PASSES}" "${SAVE_STEPS}" <<'PY'
import math, sys
steps_per_pass = int(sys.argv[1])
target_passes = float(sys.argv[2])
save_steps = int(sys.argv[3])
print(int(math.ceil(steps_per_pass * target_passes / save_steps) * save_steps))
PY
)"
fi
IFS=$'\t' read -r \
    VALID_TRAINING_WINDOWS \
    NUM_SHARDS_PER_EPOCH \
    MINIMUM_STEPS_FOR_ONE_PASS \
    COMPLETE_NOMINAL_DATA_PASSES \
    NOMINAL_DATA_PASSES < <(
        "${GROOT_PYTHON}" \
            "${GROOT_REPO}/tools/audit_franka_training_coverage.py" \
            --episodes "${LEROBOT_DATASET}/meta/episodes.jsonl" \
            --action-horizon "${ACTION_HORIZON}" \
            --shard-size "${SHARD_SIZE}" \
            --episode-sampling-rate "${EPISODE_SAMPLING_RATE}" \
            --global-batch-size "${GLOBAL_BATCH_SIZE}" \
            --max-steps "${MAX_STEPS}" \
            --output "${STATE_DIR}/frame_coverage_audit.json" \
            --format tsv
    )
if [ -z "${CHECKPOINT}" ]; then
    CHECKPOINT="${RUN_DIR}/checkpoint-${MAX_STEPS}-ema"
else
    CHECKPOINT="$(realpath -m -- "${CHECKPOINT}")"
fi
DEBUG_VIS_EPISODES="$("${GROOT_PYTHON}" - "${LEROBOT_DATASET}/meta/episodes.jsonl" "${STATE_DIR}/attention_selection.json" <<'PY'
import json, sys
from pathlib import Path
records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
preferences = [(2, 1), (2, 1), (3, 1), (3, 2)]
selected = []
used = set()
for key in preferences:
    match = next((row for row in records if row["episode_index"] not in used and row.get("blue_cube_count") == key[0] and row.get("num_preplaced") == key[1] and row["length"] > 160), None)
    if match is not None:
        selected.append(match)
        used.add(match["episode_index"])
for row in records:
    if len(selected) >= 4:
        break
    if row["episode_index"] not in used and row["length"] > 160:
        selected.append(row)
        used.add(row["episode_index"])
assert len(selected) == 4, selected
Path(sys.argv[2]).write_text(json.dumps(selected, indent=2) + "\n")
print(" ".join(str(row["episode_index"]) for row in selected))
PY
)"
status "training plan: windows=${VALID_TRAINING_WINDOWS}, steps/pass=${MINIMUM_STEPS_FOR_ONE_PASS}, max_steps=${MAX_STEPS}, nominal_passes=${NOMINAL_DATA_PASSES}, debug_episodes=${DEBUG_VIS_EPISODES}"
mark_done training_plan

CURRENT_STAGE="sft"
if [ ! -f "${STATE_DIR}/sft.done" ]; then
    if [ -d "${RUN_DIR}" ] && [ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Refusing non-empty SFT output without completion marker: ${RUN_DIR}" >&2
        exit 2
    fi
    status "starting 8-GPU SFT with EMA"
    DATASET_PATH="${LEROBOT_DATASET}" \
    VENV_PATH="${GROOT_REPO}/.venv" \
    BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
    OUTPUT_DIR="${GROOT_REPO}/outputs/franka-groot-sft" \
    HF_HOME="${HF_HOME}" \
    GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    NUM_GPUS=8 \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE}" \
    WARMUP_RATIO="${WARMUP_RATIO}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    USE_EMA=1 \
    EMA_DECAY="${EMA_DECAY}" \
    EMA_UPDATE_AFTER_STEP=0 \
    EMA_UPDATE_EVERY=1 \
    SHARD_SIZE="${SHARD_SIZE}" \
    EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE}" \
    NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH}" \
    SHORTEST_IMAGE_EDGE=256 \
    CROP_FRACTION="${CROP_FRACTION}" \
    STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB}" \
    PROCESSOR_STATE_DROPOUT_PROB="${PROCESSOR_STATE_DROPOUT_PROB}" \
    COLOR_JITTER_PARAMS="${COLOR_JITTER_PARAMS}" \
    WANDB_MODE=online \
    DEBUG_VISUALIZE=1 \
    DEBUG_VIS_EPISODES="${DEBUG_VIS_EPISODES}" \
    DEBUG_VIS_FRAME_STEP=120 \
    DEBUG_VIS_EVERY_N_CHECKPOINTS=1 \
    bash "${GROOT_REPO}/examples/Franka/train_franka.sh"
    test -f "${CHECKPOINT}/config.json"
    test -f "${CHECKPOINT}/ema_config.json"
    mark_done sft
fi

CURRENT_STAGE="ema_attention"
if [ ! -f "${STATE_DIR}/ema_attention.done" ]; then
    status "rendering four continuation-aware final EMA attention samples"
    rm -rf -- "${ATTENTION_DIR}"
    mkdir -p "${ATTENTION_DIR}"
    for episode in ${DEBUG_VIS_EPISODES}; do
        output="${ATTENTION_DIR}/final-ema-episode-${episode}-step-120.png"
        WANDB_MODE=offline "${GROOT_PYTHON}" \
            "${GROOT_REPO}/tools/visualize_franka_attention.py" \
            --dataset "${LEROBOT_DATASET}" \
            --checkpoint "${CHECKPOINT}" \
            --episode "${episode}" \
            --step 120 \
            --action-group all \
            --device cuda:0 \
            --output "${output}" \
            --wandb-project franka-groot-sft \
            --wandb-run-name "${EXPERIMENT_NAME}-ema-debug-ep${episode}" \
            --global-step "${MAX_STEPS}" \
            --full-reasoner-model "${GROOT_COSMOS_MODEL_PATH}"
        test -f "${output}"
        test -f "${output%.png}.json"
    done
    mark_done ema_attention
fi

CURRENT_STAGE="arena_eval"
if [ ! -f "${STATE_DIR}/arena_eval.done" ]; then
    if [ -d "${EVAL_OUTPUT}" ] && [ -n "$(find "${EVAL_OUTPUT}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Refusing non-empty Arena output: ${EVAL_OUTPUT}" >&2
        exit 2
    fi
    status "starting default-start 8-GPU Arena evaluation: 100 episodes/task"
    "${ARENA_REPO}/.venv/bin/python" \
        "${ARENA_REPO}/isaaclab_arena_gr00t/parallel_evaluation.py" \
        --checkpoint "${CHECKPOINT}" \
        --num-gpus 8 \
        --episodes-per-task 100 \
        --base-port 5955 \
        --arena-repo "${ARENA_REPO}" \
        --gr00t-repo "${GROOT_REPO}" \
        --cosmos-model-path "${GROOT_COSMOS_MODEL_PATH}" \
        --arena-python "${ARENA_PYTHON}" \
        --gr00t-python "${GROOT_PYTHON}" \
        --no-randomize-policy-start-pose \
        --output-dir "${EVAL_OUTPUT}"
    "${GROOT_PYTHON}" - "${EVAL_OUTPUT}/summary.json" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
assert len(summary) == 3, summary
for task, result in summary.items():
    assert result["episodes"] == 100, (task, result)
print(json.dumps(summary, indent=2))
PY
    test -f "${EVAL_OUTPUT}/index.html"
    mark_done arena_eval
fi

CURRENT_STAGE="cleanup"
if [ ! -f "${STATE_DIR}/cleanup.done" ]; then
    status "pruning non-final SFT checkpoints; preserving final EMA only"
    "${GROOT_PYTHON}" - "${RUN_DIR}" "${CHECKPOINT}" <<'PY'
import shutil, sys
from pathlib import Path
run_dir = Path(sys.argv[1]).resolve()
keep = Path(sys.argv[2]).resolve()
for path in run_dir.glob("checkpoint-*"):
    if path.resolve() != keep:
        shutil.rmtree(path)
PY
    mark_done cleanup
fi

CURRENT_STAGE="complete"
mark_done complete
status "FINAL FRANKA PIPELINE COMPLETE"
