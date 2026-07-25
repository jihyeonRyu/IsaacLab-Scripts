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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-fixedtray-waypoint10-recovery1-v5}"
CHECKPOINT="${CHECKPOINT:-}"
EVAL_OUTPUT="${EVAL_OUTPUT:-}"
STATE_DIR="${STATE_DIR:-}"
GENERATION_SEED="${GENERATION_SEED:-70007}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
PRINT_CONFIG=0

usage() {
    cat <<'EOF'
Run the v5 Franka 8-GPU E2E ablation: generate → analyze → convert → SFT → Arena.

Usage:
  run_v5_waypoint10_recovery1.sh [options]

Common path options:
  --workspace-root PATH  Parent used for all unspecified paths (default: /workspace)
  --scripts-repo PATH    IsaacLab-Scripts checkout
  --groot-repo PATH      Isaac-GR00T checkout
  --arena-repo PATH      IsaacLab-Arena checkout
  --models-root PATH     Parent containing both local models and HF cache
  --isaac-python PATH    Isaac generation/Arena-worker Python
  --arena-python PATH    Arena-worker Python (default: ISAAC_PYTHON)
  --groot-python PATH    GR00T conversion/server Python

Output overrides:
  --raw-dataset PATH
  --lerobot-dataset PATH
  --experiment-name NAME
  --checkpoint PATH
  --eval-output PATH
  --state-dir PATH
  --generation-seed N
  --global-batch-size N Global SFT batch; must be divisible by 8 (default: 128)
  --wait-for-pid PID

Advanced model overrides:
  --base-model-path PATH
  --cosmos-model-path PATH
  --hf-home PATH
  --ffmpeg-runtime PATH
  --groot-branch NAME
  --arena-branch NAME

Other:
  --print-config         Resolve and print every path without starting work
  -h, --help             Show this help

Fixed v5 experiment settings:
  600 attempts, seed 70007 by default, waypoint probability 0.10,
  waypoint radius 4–8 cm, and one solver-recovery attempt per cube.
  GR00T predicts 40 actions; Arena executes 16 actions at the dataset's 15 Hz.
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
        --experiment-name) need_value "$@"; EXPERIMENT_NAME=$2; shift 2 ;;
        --checkpoint) need_value "$@"; CHECKPOINT=$2; shift 2 ;;
        --eval-output) need_value "$@"; EVAL_OUTPUT=$2; shift 2 ;;
        --state-dir) need_value "$@"; STATE_DIR=$2; shift 2 ;;
        --generation-seed) need_value "$@"; GENERATION_SEED=$2; shift 2 ;;
        --global-batch-size) need_value "$@"; GLOBAL_BATCH_SIZE=$2; shift 2 ;;
        --wait-for-pid) need_value "$@"; WAIT_FOR_PID=$2; shift 2 ;;
        --groot-branch) need_value "$@"; GROOT_BRANCH=$2; shift 2 ;;
        --arena-branch) need_value "$@"; ARENA_BRANCH=$2; shift 2 ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "${GENERATION_SEED}" =~ ^[0-9]+$ ]]; then
    echo "--generation-seed must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || [ "$((GLOBAL_BATCH_SIZE % 8))" -ne 0 ]; then
    echo "--global-batch-size must be a positive integer divisible by 8" >&2
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
ISAAC_PYTHON="$(normalize_executable_path "${ISAAC_PYTHON:-${WORKSPACE_ROOT}/env_isaaclab/bin/python}")"
ARENA_PYTHON="$(normalize_executable_path "${ARENA_PYTHON:-${ISAAC_PYTHON}}")"
GROOT_PYTHON="$(normalize_executable_path "${GROOT_PYTHON:-${GROOT_REPO}/.venv/bin/python}")"
RAW_DATASET="$(realpath -m -- "${RAW_DATASET:-${WORKSPACE_ROOT}/output/franka_waypoint10_recovery1_600eps_seed${GENERATION_SEED}_v5}")"
LEROBOT_DATASET="$(realpath -m -- "${LEROBOT_DATASET:-${WORKSPACE_ROOT}/datasets/franka_waypoint10_recovery1_seed${GENERATION_SEED}_v5_lerobot}")"
CHECKPOINT="$(realpath -m -- "${CHECKPOINT:-${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}/checkpoint-10000}")"
EVAL_OUTPUT="$(realpath -m -- "${EVAL_OUTPUT:-${ARENA_REPO}/outputs/franka-gr00t-parallel/fixedtray-waypoint10-recovery1-v5-generation-aligned-8gpu-100eps}")"
STATE_DIR="$(realpath -m -- "${STATE_DIR:-${WORKSPACE_ROOT}/output/franka_e2e_pipeline_waypoint10_recovery1_v5}")"

print_config() {
    cat <<EOF
Resolved v5 E2E paths
  workspace root : ${WORKSPACE_ROOT}
  scripts repo   : ${SCRIPTS_REPO}
  GR00T repo     : ${GROOT_REPO}
  Arena repo     : ${ARENA_REPO}
  models root    : ${MODELS_ROOT}
  GR00T model    : ${BASE_MODEL_PATH}
  Cosmos model   : ${GROOT_COSMOS_MODEL_PATH}
  HF cache       : ${HF_HOME}
  FFmpeg runtime : ${FFMPEG_RUNTIME}
  Isaac Python   : ${ISAAC_PYTHON}
  Arena Python   : ${ARENA_PYTHON}
  GR00T Python   : ${GROOT_PYTHON}
  raw dataset    : ${RAW_DATASET}
  LeRobot output : ${LEROBOT_DATASET}
  experiment     : ${EXPERIMENT_NAME}
  checkpoint     : ${CHECKPOINT}
  Arena output   : ${EVAL_OUTPUT}
  state dir      : ${STATE_DIR}
  generation seed: ${GENERATION_SEED}
  SFT global batch: ${GLOBAL_BATCH_SIZE} ($((GLOBAL_BATCH_SIZE / 8)) per GPU)
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
    "${SCRIPTS_REPO}/franka_groot_e2e/run_pipeline.sh" \
    "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/franka_lift_auto_parallel.py" \
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
if ! "${GROOT_REPO}/.venv/bin/wandb" login --verify >/dev/null 2>&1; then
    echo "W&B login verification failed. Run: ${GROOT_REPO}/.venv/bin/wandb login" >&2
    exit 3
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
LOCAL_ISAAC_LIB="${WORKSPACE_ROOT}/.tools/isaac-system-libs/usr/lib/x86_64-linux-gnu"
if [ -d "${LOCAL_ISAAC_LIB}" ]; then
    export LD_LIBRARY_PATH="${LOCAL_ISAAC_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONUNBUFFERED=1

if [ -n "${WAIT_FOR_PID}" ]; then
    printf '[%s] waiting for PID %s before v5 generation\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${WAIT_FOR_PID}"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
fi

if [ ! -f "${RAW_DATASET}/multi_gpu_summary.json" ]; then
    if [ -d "${RAW_DATASET}" ] && [ -n "$(find "${RAW_DATASET}" -mindepth 1 -print -quit)" ]; then
        echo "Refusing to reuse incomplete non-empty raw dataset: ${RAW_DATASET}" >&2
        exit 2
    fi

    "${ISAAC_PYTHON}" \
        "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/franka_lift_auto_parallel.py" \
        --headless \
        --enable_cameras \
        --num_envs 4 \
        --auto_generate_episodes 600 \
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
        --solver_recovery_max_attempts 1
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
SCRIPTS_REPO="${SCRIPTS_REPO}" \
GROOT_REPO="${GROOT_REPO}" \
ARENA_REPO="${ARENA_REPO}" \
MODELS_ROOT="${MODELS_ROOT}" \
BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH}" \
HF_HOME="${HF_HOME}" \
ISAAC_PYTHON="${ISAAC_PYTHON}" \
ARENA_PYTHON="${ARENA_PYTHON}" \
GROOT_PYTHON="${GROOT_PYTHON}" \
RAW_DATASET="${RAW_DATASET}" \
LEROBOT_DATASET="${LEROBOT_DATASET}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT="${CHECKPOINT}" \
EVAL_OUTPUT="${EVAL_OUTPUT}" \
EXPECTED_EPISODES=600 \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
STATE_DIR="${STATE_DIR}" \
GROOT_BRANCH="${GROOT_BRANCH}" \
ARENA_BRANCH="${ARENA_BRANCH}" \
exec bash "${SCRIPTS_REPO}/franka_groot_e2e/run_pipeline.sh"
