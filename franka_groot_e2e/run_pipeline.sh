#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
GROOT_REPO="${GROOT_REPO:-${WORKSPACE_ROOT}/Isaac-GR00T}"
ARENA_REPO="${ARENA_REPO:-${WORKSPACE_ROOT}/IsaacLab-Arena}"
SCRIPTS_REPO="${SCRIPTS_REPO:-${WORKSPACE_ROOT}/IsaacLab-Scripts}"
MODELS_ROOT="${MODELS_ROOT:-${WORKSPACE_ROOT}/models}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_ROOT}/GR00T-N1.7-3B}"
HF_HOME="${HF_HOME:-${MODELS_ROOT}/huggingface-cache}"
GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH:-${MODELS_ROOT}/Cosmos-Reason2-2B}"
FFMPEG_RUNTIME="${FFMPEG_RUNTIME:-${WORKSPACE_ROOT}/.tools/ffmpeg-7}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${WORKSPACE_ROOT}/env_isaaclab/bin/python}"
ARENA_PYTHON="${ARENA_PYTHON:-${ISAAC_PYTHON}}"
GROOT_PYTHON="${GROOT_PYTHON:-${GROOT_REPO}/.venv/bin/python}"
RAW_DATASET="${RAW_DATASET:-${WORKSPACE_ROOT}/output/franka_posture_recovery_rgbfull_600eps_seed50007_20260724}"
LEROBOT_DATASET="${LEROBOT_DATASET:-${WORKSPACE_ROOT}/datasets/franka_posture_recovery_rgbfull_seed50007_lerobot}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-fixedtray-recovery-v3}"
CHECKPOINT="${CHECKPOINT:-${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}/checkpoint-10000}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${ARENA_REPO}/outputs/franka-gr00t-parallel/fixedtray-recovery-v3-8gpu-100eps}"
GENERATION_PID="${GENERATION_PID:-}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-600}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
STATE_DIR="${STATE_DIR:-${WORKSPACE_ROOT}/output/franka_e2e_pipeline_fixedtray_recovery_v3}"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/pipeline.lock"
flock -n 9 || {
    echo "Another Franka E2E supervisor already holds ${STATE_DIR}/pipeline.lock" >&2
    exit 1
}

status() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${STATE_DIR}/status.log"
}

require_branch() {
    local repo_path=$1
    local expected_branch=$2
    local actual_branch
    actual_branch="$(git -C "${repo_path}" branch --show-current)"
    if [ "${actual_branch}" != "${expected_branch}" ]; then
        echo "Expected ${repo_path} on branch ${expected_branch}, found ${actual_branch:-detached HEAD}" >&2
        exit 2
    fi
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

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
LOCAL_ISAAC_LIB="${WORKSPACE_ROOT}/.tools/isaac-system-libs/usr/lib/x86_64-linux-gnu"
if [ -d "${LOCAL_ISAAC_LIB}" ]; then
    export LD_LIBRARY_PATH="${LOCAL_ISAAC_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [ ! -x "${FFMPEG_RUNTIME}/bin/ffmpeg" ]; then
    echo "FFmpeg runtime is missing: ${FFMPEG_RUNTIME}/bin/ffmpeg" >&2
    exit 2
fi
export PATH="${FFMPEG_RUNTIME}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${FFMPEG_RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

require_branch "${GROOT_REPO}" "${GROOT_BRANCH}"
require_branch "${ARENA_REPO}" "${ARENA_BRANCH}"
for required_path in \
    "${ISAAC_PYTHON}" \
    "${ARENA_PYTHON}" \
    "${GROOT_PYTHON}" \
    "${BASE_MODEL_PATH}/config.json" \
    "${GROOT_COSMOS_MODEL_PATH}/config.json"; do
    if [ ! -e "${required_path}" ]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done
if ! "${GROOT_PYTHON}" -c 'from torchcodec.decoders import VideoDecoder' >/dev/null; then
    echo "TorchCodec cannot load the FFmpeg runtime at ${FFMPEG_RUNTIME}." >&2
    exit 2
fi

CURRENT_STAGE="generation"
if [ ! -f "${STATE_DIR}/generation.done" ]; then
    status "waiting for 8-GPU generation"
    if [ -n "${GENERATION_PID}" ]; then
        while kill -0 "${GENERATION_PID}" 2>/dev/null; do
            sleep 60
        done
    else
        while pgrep -f "franka_lift_auto_parallel.py.*${RAW_DATASET}" >/dev/null; do
            sleep 60
        done
    fi
    "${ISAAC_PYTHON}" - "${RAW_DATASET}/multi_gpu_summary.json" "${EXPECTED_EPISODES}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = int(sys.argv[2])
summary = json.loads(summary_path.read_text())
assert summary["requested_episodes"] == expected, summary
assert summary["reported_episodes"] == expected, summary
assert summary["all_workers_exited_cleanly"] is True, summary
print(json.dumps(summary, indent=2))
PY
    mark_done generation
fi

CURRENT_STAGE="analysis"
if [ ! -f "${STATE_DIR}/analysis.done" ]; then
    status "analyzing generated trajectories"
    "${ISAAC_PYTHON}" \
        "${SCRIPTS_REPO}/franka_groot_e2e/scripts/01_generate/analyze_franka_trajectories.py" \
        "${RAW_DATASET}" \
        --output-dir "${RAW_DATASET}/trajectory_analysis"
    mark_done analysis
fi

CURRENT_STAGE="conversion"
if [ ! -f "${STATE_DIR}/conversion.done" ]; then
    status "converting successful episodes to LeRobot v2.1"
    if [ -d "${LEROBOT_DATASET}" ] && [ -n "$(find "${LEROBOT_DATASET}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "LeRobot output is already non-empty without a completion marker: ${LEROBOT_DATASET}" >&2
        exit 2
    fi
    "${GROOT_PYTHON}" \
        "${SCRIPTS_REPO}/franka_groot_e2e/scripts/02_convert/convert_franka_to_groot_lerobot.py" \
        "${RAW_DATASET}" \
        "${LEROBOT_DATASET}"
    test -f "${LEROBOT_DATASET}/meta/info.json"
    mark_done conversion
fi

CURRENT_STAGE="sft"
if [ ! -f "${STATE_DIR}/sft.done" ]; then
    status "starting 8-GPU GR00T SFT"
    DATASET_PATH="${LEROBOT_DATASET}" \
    VENV_PATH="${GROOT_REPO}/.venv" \
    BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
    OUTPUT_DIR="${GROOT_REPO}/outputs/franka-groot-sft" \
    HF_HOME="${HF_HOME}" \
    GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    NUM_GPUS=8 \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    MAX_STEPS=10000 \
    SAVE_STEPS=250 \
    WANDB_MODE=online \
    DEBUG_VISUALIZE=1 \
    bash "${GROOT_REPO}/examples/Franka/train_franka.sh"
    test -f "${CHECKPOINT}/config.json"
    mark_done sft
fi

CURRENT_STAGE="arena_eval"
if [ ! -f "${STATE_DIR}/arena_eval.done" ]; then
    status "starting 8-GPU Arena evaluation: 100 episodes per cube count"
    "${ARENA_REPO}/.venv/bin/python" \
        "${ARENA_REPO}/isaaclab_arena_gr00t/parallel_evaluation.py" \
        --checkpoint "${CHECKPOINT}" \
        --num-gpus 8 \
        --episodes-per-task 100 \
        --base-port 5655 \
        --arena-repo "${ARENA_REPO}" \
        --gr00t-repo "${GROOT_REPO}" \
        --cosmos-model-path "${GROOT_COSMOS_MODEL_PATH}" \
        --arena-python "${ARENA_PYTHON}" \
        --gr00t-python "${GROOT_PYTHON}" \
        --output-dir "${EVAL_OUTPUT}"
    test -f "${EVAL_OUTPUT}/summary.json"
    test -f "${EVAL_OUTPUT}/index.html"
    mark_done arena_eval
fi

CURRENT_STAGE="complete"
mark_done complete
status "CORE PIPELINE COMPLETE; outputs ready for final asset and README curation"
