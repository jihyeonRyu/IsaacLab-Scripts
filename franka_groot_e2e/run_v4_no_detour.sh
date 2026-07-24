#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPTS_REPO="${SCRIPTS_REPO:-/workspace/IsaacLab-Scripts}"
RAW_DATASET="${RAW_DATASET:-/workspace/output/franka_no_detour_posture_recovery_600eps_seed60007_v4}"
LEROBOT_DATASET="${LEROBOT_DATASET:-/workspace/datasets/franka_no_detour_posture_recovery_seed60007_v4_lerobot}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-fixedtray-no-detour-recovery-v4}"
EVAL_OUTPUT="${EVAL_OUTPUT:-/workspace/IsaacLab-Arena/outputs/franka-gr00t-parallel/fixedtray-no-detour-recovery-v4-generation-aligned-8gpu-100eps}"
STATE_DIR="${STATE_DIR:-/workspace/output/franka_e2e_pipeline_no_detour_recovery_v4}"
GENERATION_SEED="${GENERATION_SEED:-60007}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

if [ -n "${WAIT_FOR_PID}" ]; then
    printf '[%s] waiting for PID %s before v4 generation\n' \
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

    /workspace/env_isaaclab/bin/python \
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
        --randomize_start_pose \
        --start_ee_x_range 0.36 0.70 \
        --start_ee_y_range -0.34 0.34 \
        --start_ee_z_range 0.25 0.55 \
        --start_ee_radius_min 0.40 \
        --start_ee_radius_max 0.72 \
        --recovery_waypoint_prob 0.0 \
        --solver_recovery_max_attempts 3
fi

RAW_DATASET="${RAW_DATASET}" \
LEROBOT_DATASET="${LEROBOT_DATASET}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT="/workspace/Isaac-GR00T/outputs/franka-groot-sft/${EXPERIMENT_NAME}/checkpoint-10000" \
EVAL_OUTPUT="${EVAL_OUTPUT}" \
EXPECTED_EPISODES=600 \
STATE_DIR="${STATE_DIR}" \
exec bash "${SCRIPTS_REPO}/franka_groot_e2e/run_pipeline.sh"
