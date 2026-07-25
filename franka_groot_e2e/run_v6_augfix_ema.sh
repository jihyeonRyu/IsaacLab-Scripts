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
ARENA_PYTHON="${ARENA_PYTHON:-}"
GROOT_PYTHON="${GROOT_PYTHON:-}"
LEROBOT_DATASET="${LEROBOT_DATASET:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-fixedtray-waypoint10-recovery1-v6-augfix-ema}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}"
STATE_DIR="${STATE_DIR:-}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
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
PRINT_CONFIG=0

usage() {
    cat <<'EOF'
Train the corrected Franka augmentation configuration with EMA, then evaluate it.

This reuses the completed v5 LeRobot dataset. It does not regenerate or convert
data. The launcher can wait for another GPU job before starting.

Usage:
  run_v6_augfix_ema.sh [options]

Path options:
  --workspace-root PATH
  --scripts-repo PATH
  --groot-repo PATH
  --arena-repo PATH
  --models-root PATH
  --base-model-path PATH
  --cosmos-model-path PATH
  --hf-home PATH
  --ffmpeg-runtime PATH
  --arena-python PATH
  --groot-python PATH
  --lerobot-dataset PATH
  --experiment-name NAME
  --checkpoint PATH
  --eval-output PATH
  --state-dir PATH

Training options:
  --global-batch-size N
  --max-steps N
  --learning-rate VALUE
  --ema-decay VALUE
  --wait-for-pid PID

Other:
  --groot-branch NAME
  --arena-branch NAME
  --print-config
  -h, --help

Defaults:
  batch 128, 10,000 steps, LR 1e-4 cosine with 5% warmup, crop 0.98,
  action-head state dropout 0.2, processor state dropout 0.0, EMA 0.999,
  model horizon 40, and Arena execution horizon 16.
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
        --arena-python) need_value "$@"; ARENA_PYTHON=$2; shift 2 ;;
        --groot-python) need_value "$@"; GROOT_PYTHON=$2; shift 2 ;;
        --lerobot-dataset) need_value "$@"; LEROBOT_DATASET=$2; shift 2 ;;
        --experiment-name) need_value "$@"; EXPERIMENT_NAME=$2; shift 2 ;;
        --checkpoint) need_value "$@"; CHECKPOINT=$2; shift 2 ;;
        --eval-output) need_value "$@"; EVAL_OUTPUT=$2; shift 2 ;;
        --state-dir) need_value "$@"; STATE_DIR=$2; shift 2 ;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE=$2; shift 2 ;;
        --max-steps) need_value "$@"; MAX_STEPS=$2; shift 2 ;;
        --learning-rate) need_value "$@"; LEARNING_RATE=$2; shift 2 ;;
        --ema-decay) need_value "$@"; EMA_DECAY=$2; shift 2 ;;
        --wait-for-pid) need_value "$@"; WAIT_FOR_PID=$2; shift 2 ;;
        --groot-branch) need_value "$@"; GROOT_BRANCH=$2; shift 2 ;;
        --arena-branch) need_value "$@"; ARENA_BRANCH=$2; shift 2 ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in GLOBAL_BATCH_SIZE MAX_STEPS SAVE_STEPS ACTION_HORIZON; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done
if [ "$((GLOBAL_BATCH_SIZE % 8))" -ne 0 ]; then
    echo "GLOBAL_BATCH_SIZE must be divisible by 8" >&2
    exit 2
fi
if [ -n "${WAIT_FOR_PID}" ] && ! [[ "${WAIT_FOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--wait-for-pid must be a positive integer" >&2
    exit 2
fi

WORKSPACE_ROOT="$(realpath -m -- "${WORKSPACE_ROOT}")"
SCRIPTS_REPO="$(realpath -m -- "${SCRIPTS_REPO:-${DEFAULT_SCRIPTS_REPO}}")"
GROOT_REPO="$(realpath -m -- "${GROOT_REPO:-${WORKSPACE_ROOT}/Isaac-GR00T}")"
ARENA_REPO="$(realpath -m -- "${ARENA_REPO:-${WORKSPACE_ROOT}/IsaacLab-Arena}")"
MODELS_ROOT="$(realpath -m -- "${MODELS_ROOT:-${WORKSPACE_ROOT}/models}")"
BASE_MODEL_PATH="$(realpath -m -- "${BASE_MODEL_PATH:-${MODELS_ROOT}/GR00T-N1.7-3B}")"
GROOT_COSMOS_MODEL_PATH="$(realpath -m -- "${GROOT_COSMOS_MODEL_PATH:-${MODELS_ROOT}/Cosmos-Reason2-2B}")"
HF_HOME="$(realpath -m -- "${HF_HOME:-${MODELS_ROOT}/huggingface-cache}")"
FFMPEG_RUNTIME="$(realpath -m -- "${FFMPEG_RUNTIME:-${WORKSPACE_ROOT}/.tools/ffmpeg-7}")"
ARENA_PYTHON="$(normalize_executable_path "${ARENA_PYTHON:-${WORKSPACE_ROOT}/env_isaaclab/bin/python}")"
GROOT_PYTHON="$(normalize_executable_path "${GROOT_PYTHON:-${GROOT_REPO}/.venv/bin/python}")"
LEROBOT_DATASET="$(realpath -m -- "${LEROBOT_DATASET:-${WORKSPACE_ROOT}/datasets/franka_waypoint10_recovery1_seed70007_v5_lerobot}")"
CHECKPOINT="$(realpath -m -- "${CHECKPOINT:-${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}/checkpoint-${MAX_STEPS}-ema}")"
EVAL_OUTPUT="$(realpath -m -- "${EVAL_OUTPUT:-${ARENA_REPO}/outputs/franka-gr00t-parallel/fixedtray-waypoint10-recovery1-v6-augfix-ema-generation-aligned-8gpu-100eps}")"
STATE_DIR="$(realpath -m -- "${STATE_DIR:-${WORKSPACE_ROOT}/output/franka_e2e_pipeline_waypoint10_recovery1_v6_augfix_ema}")"
RUN_DIR="${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}"
ATTENTION_DIR="${GROOT_REPO}/outputs/attention/${EXPERIMENT_NAME}"

print_config() {
    cat <<EOF
Resolved v6 augmentation-fix + EMA paths/settings
  workspace root : ${WORKSPACE_ROOT}
  scripts repo   : ${SCRIPTS_REPO}
  GR00T repo     : ${GROOT_REPO}
  Arena repo     : ${ARENA_REPO}
  GR00T model    : ${BASE_MODEL_PATH}
  Cosmos model   : ${GROOT_COSMOS_MODEL_PATH}
  LeRobot input  : ${LEROBOT_DATASET}
  FFmpeg runtime : ${FFMPEG_RUNTIME}
  experiment     : ${EXPERIMENT_NAME}
  EMA checkpoint : ${CHECKPOINT}
  attention dir  : ${ATTENTION_DIR}
  Arena output   : ${EVAL_OUTPUT}
  state dir      : ${STATE_DIR}
  batch/steps    : ${GLOBAL_BATCH_SIZE}/${MAX_STEPS} ($((GLOBAL_BATCH_SIZE / 8)) per GPU)
  frame sharding : size=${SHARD_SIZE}; episode split rate=${EPISODE_SAMPLING_RATE}
  LR schedule    : ${LEARNING_RATE} ${LR_SCHEDULER_TYPE}, warmup ${WARMUP_RATIO}
  augmentation   : crop=${CROP_FRACTION}; jitter=${COLOR_JITTER_PARAMS}
  state dropout  : action head=${STATE_DROPOUT_PROB}; processor=${PROCESSOR_STATE_DROPOUT_PROB}
  horizon        : train=${ACTION_HORIZON}; Arena execute=16
  EMA decay      : ${EMA_DECAY}
EOF
}

print_config
if [ "${PRINT_CONFIG}" = "1" ]; then
    exit 0
fi

for required_path in \
    "${ARENA_PYTHON}" \
    "${GROOT_PYTHON}" \
    "${BASE_MODEL_PATH}/config.json" \
    "${GROOT_COSMOS_MODEL_PATH}/config.json" \
    "${LEROBOT_DATASET}/meta/info.json" \
    "${LEROBOT_DATASET}/meta/episodes.jsonl" \
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
    echo "Run install_franka_groot_e2e.sh or pass --ffmpeg-runtime." >&2
    exit 2
fi
export PATH="${FFMPEG_RUNTIME}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${FFMPEG_RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if ! "${GROOT_PYTHON}" -c 'from torchcodec.decoders import VideoDecoder' >/dev/null; then
    echo "TorchCodec cannot load the FFmpeg runtime at ${FFMPEG_RUNTIME}." >&2
    exit 2
fi
if ! grep -q -- '--processor-state-dropout-prob' "${GROOT_REPO}/examples/Franka/train_franka.sh"; then
    echo "GR00T checkout lacks the processor augmentation override fix" >&2
    exit 2
fi
if ! grep -q -- 'USE_EMA' "${GROOT_REPO}/examples/Franka/train_franka.sh"; then
    echo "GR00T checkout lacks EMA support" >&2
    exit 2
fi
if ! "${GROOT_REPO}/.venv/bin/wandb" login --verify >/dev/null 2>&1; then
    echo "W&B login verification failed. Run: ${GROOT_REPO}/.venv/bin/wandb login" >&2
    exit 3
fi

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/pipeline.lock"
flock -n 9 || {
    echo "Another v6 supervisor already holds ${STATE_DIR}/pipeline.lock" >&2
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

export HF_HOME
export GROOT_COSMOS_MODEL_PATH
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export PYTHONUNBUFFERED=1

if [ -n "${WAIT_FOR_PID}" ]; then
    status "waiting for PID ${WAIT_FOR_PID} before corrected SFT"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
fi

CURRENT_STAGE="data_pass_audit"
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
status "frame coverage: windows=${VALID_TRAINING_WINDOWS}, shards=${NUM_SHARDS_PER_EPOCH}, steps/pass=${MINIMUM_STEPS_FOR_ONE_PASS}, complete/nominal passes=${COMPLETE_NOMINAL_DATA_PASSES}/${NOMINAL_DATA_PASSES}"
mark_done data_pass_audit

CURRENT_STAGE="sft"
if [ ! -f "${STATE_DIR}/sft.done" ]; then
    if [ -d "${RUN_DIR}" ] && [ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Refusing to overwrite non-empty training output without sft.done: ${RUN_DIR}" >&2
        exit 2
    fi
    status "starting corrected 8-GPU SFT with FP32 EMA"
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
    DEBUG_VIS_EPISODES="0 1 2 3" \
    bash "${GROOT_REPO}/examples/Franka/train_franka.sh"
    test -f "${CHECKPOINT}/config.json"
    test -f "${CHECKPOINT}/ema_config.json"
    mark_done sft
fi

CURRENT_STAGE="ema_attention"
if [ ! -f "${STATE_DIR}/ema_attention.done" ]; then
    status "rendering four EMA-checkpoint attention samples"
    mkdir -p "${ATTENTION_DIR}"
    for episode in 0 1 2 3; do
        output="${ATTENTION_DIR}/checkpoint-${MAX_STEPS}-ema-ep${episode}-step120.png"
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
        echo "Refusing to overwrite non-empty Arena output without arena_eval.done: ${EVAL_OUTPUT}" >&2
        exit 2
    fi
    status "starting EMA Arena evaluation: 8 GPUs, 100 episodes per cube count"
    "${ARENA_REPO}/.venv/bin/python" \
        "${ARENA_REPO}/isaaclab_arena_gr00t/parallel_evaluation.py" \
        --checkpoint "${CHECKPOINT}" \
        --num-gpus 8 \
        --episodes-per-task 100 \
        --base-port 5755 \
        --arena-repo "${ARENA_REPO}" \
        --gr00t-repo "${GROOT_REPO}" \
        --cosmos-model-path "${GROOT_COSMOS_MODEL_PATH}" \
        --arena-python "${ARENA_PYTHON}" \
        --gr00t-python "${GROOT_PYTHON}" \
        --output-dir "${EVAL_OUTPUT}"
    "${GROOT_PYTHON}" - "${EVAL_OUTPUT}/summary.json" <<'PY'
import json
import sys
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

CURRENT_STAGE="complete"
mark_done complete
status "V6 AUGMENTATION-FIX + EMA PIPELINE COMPLETE"
