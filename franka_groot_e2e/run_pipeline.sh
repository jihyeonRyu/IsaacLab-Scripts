#!/usr/bin/env bash

set -Eeuo pipefail

RAW_DATASET="${RAW_DATASET:-/workspace/output/franka_posture_recovery_rgbfull_600eps_seed50007_20260724}"
LEROBOT_DATASET="${LEROBOT_DATASET:-/workspace/datasets/franka_posture_recovery_rgbfull_seed50007_lerobot}"
GROOT_REPO="${GROOT_REPO:-/workspace/Isaac-GR00T}"
ARENA_REPO="${ARENA_REPO:-/workspace/IsaacLab-Arena}"
SCRIPTS_REPO="${SCRIPTS_REPO:-/workspace/IsaacLab-Scripts}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-fixedtray-recovery-v3}"
CHECKPOINT="${CHECKPOINT:-${GROOT_REPO}/outputs/franka-groot-sft/${EXPERIMENT_NAME}/checkpoint-10000}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${ARENA_REPO}/outputs/franka-gr00t-parallel/fixedtray-recovery-v3-8gpu-100eps}"
GENERATION_PID="${GENERATION_PID:-}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-600}"
STATE_DIR="${STATE_DIR:-/workspace/output/franka_e2e_pipeline_fixedtray_recovery_v3}"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/pipeline.lock"
flock -n 9 || {
    echo "Another Franka E2E supervisor already holds ${STATE_DIR}/pipeline.lock" >&2
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

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export LD_LIBRARY_PATH="/workspace/.tools/isaac-system-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

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
    /workspace/env_isaaclab/bin/python - "${RAW_DATASET}/multi_gpu_summary.json" "${EXPECTED_EPISODES}" <<'PY'
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
    /workspace/env_isaaclab/bin/python \
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
    "${GROOT_REPO}/.venv/bin/python" \
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
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    NUM_GPUS=8 \
    GLOBAL_BATCH_SIZE=64 \
    MAX_STEPS=10000 \
    SAVE_STEPS=250 \
    WANDB_MODE=online \
    DEBUG_VISUALIZE=1 \
    bash "${SCRIPTS_REPO}/franka_groot_e2e/scripts/03_sft/train_franka.sh"
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
        --output-dir "${EVAL_OUTPUT}"
    test -f "${EVAL_OUTPUT}/summary.json"
    test -f "${EVAL_OUTPUT}/index.html"
    mark_done arena_eval
fi

CURRENT_STAGE="complete"
mark_done complete
status "CORE PIPELINE COMPLETE; outputs ready for final asset and README curation"
