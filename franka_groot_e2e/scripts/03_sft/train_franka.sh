#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
E2E_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GROOT_REPO="${GROOT_REPO:-/workspace/Isaac-GR00T}"

VENV_PATH="${VENV_PATH:-${GROOT_REPO}/.venv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/models/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-/workspace/datasets/franka_parallel_groot_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${GROOT_REPO}/outputs/franka-groot-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-franka-blue-cube-sft-crop098-aug-v2}"
WANDB_PROJECT="${WANDB_PROJECT:-franka-gr00t}"
HF_HOME="${HF_HOME:-/workspace/models/huggingface-cache}"
GROOT_COSMOS_MODEL_PATH="${GROOT_COSMOS_MODEL_PATH:-/workspace/models/Cosmos-Reason2-2B}"

NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SHARD_SIZE="${SHARD_SIZE:-512}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-1.0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
TUNE_LLM="${TUNE_LLM:-0}"
SHORTEST_IMAGE_EDGE="${SHORTEST_IMAGE_EDGE:-256}"
CROP_FRACTION="${CROP_FRACTION:-0.98}"
STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"
COLOR_JITTER_PARAMS="${COLOR_JITTER_PARAMS:-brightness 0.25 contrast 0.25 saturation 0.30 hue 0.03}"
DEBUG_VISUALIZE="${DEBUG_VISUALIZE:-1}"
DEBUG_VIS_EPISODES="${DEBUG_VIS_EPISODES:-0 1 2 3}"
DEBUG_VIS_FRAME_STEP="${DEBUG_VIS_FRAME_STEP:-120}"
DEBUG_VIS_ACTION_GROUP="${DEBUG_VIS_ACTION_GROUP:-all}"
DEBUG_VIS_DEVICE="${DEBUG_VIS_DEVICE:-cuda:0}"
DEBUG_VIS_MIN_STEP="${DEBUG_VIS_MIN_STEP:-${SAVE_STEPS}}"
DEBUG_VIS_EVERY_N_CHECKPOINTS="${DEBUG_VIS_EVERY_N_CHECKPOINTS:-1}"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-${GROOT_REPO}/outputs/attention/${EXPERIMENT_NAME}}"

for path in "${VENV_PATH}/bin/activate" "${BASE_MODEL_PATH}" "${GROOT_COSMOS_MODEL_PATH}" "${DATASET_PATH}"; do
    if [ ! -e "${path}" ]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export HF_HOME GROOT_COSMOS_MODEL_PATH WANDB_PROJECT WANDB_MODE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

if [ ! -f "${GROOT_COSMOS_MODEL_PATH}/config.json" ]; then
    echo "Cosmos config is missing: ${GROOT_COSMOS_MODEL_PATH}/config.json" >&2
    exit 2
fi

if [ "${USE_WANDB}" = "1" ] && ! wandb login --verify >/dev/null 2>&1; then
    echo "W&B login verification failed. Run: wandb login" >&2
    exit 3
fi

DEBUG_VIS_EPISODE_ARGS=()
read -r -a DEBUG_VIS_EPISODE_VALUES <<< "${DEBUG_VIS_EPISODES}"
for episode in "${DEBUG_VIS_EPISODE_VALUES[@]}"; do
    if ! [[ "${episode}" =~ ^[0-9]+$ ]]; then
        echo "Invalid debug episode index: ${episode}" >&2
        exit 4
    fi
    DEBUG_VIS_EPISODE_ARGS+=(--episode "${episode}")
done
if [ "${#DEBUG_VIS_EPISODE_ARGS[@]}" -eq 0 ]; then
    echo "DEBUG_VIS_EPISODES must contain at least one episode index" >&2
    exit 4
fi

EXTRA_ARGS=()
if [ "${TUNE_LLM}" = "1" ]; then
    EXTRA_ARGS=(-- --tune-llm)
fi

IMAGE_AUG_ARGS=()
if [ -n "${SHORTEST_IMAGE_EDGE}" ]; then
    IMAGE_AUG_ARGS+=(--shortest-image-edge "${SHORTEST_IMAGE_EDGE}")
fi
if [ -n "${CROP_FRACTION}" ]; then
    IMAGE_AUG_ARGS+=(--crop-fraction "${CROP_FRACTION}")
fi

echo "Starting Franka GR00T fine-tuning"
echo "  e2e package: ${E2E_ROOT}"
echo "  dataset: ${DATASET_PATH}"
echo "  base model: ${BASE_MODEL_PATH}"
echo "  output: ${OUTPUT_DIR}/${EXPERIMENT_NAME}"
echo "  W&B: ${WANDB_PROJECT}/${EXPERIMENT_NAME} (${WANDB_MODE})"
echo "  GPUs/global batch/steps: ${NUM_GPUS}/${GLOBAL_BATCH_SIZE}/${MAX_STEPS}"
echo "  shortest edge/crop fraction: ${SHORTEST_IMAGE_EDGE:-legacy}/${CROP_FRACTION:-legacy}"
echo "  state dropout: ${STATE_DROPOUT_PROB}"
echo "  color jitter: ${COLOR_JITTER_PARAMS}"
echo "  tune reasoner: ${TUNE_LLM}"
echo "  automatic offline W&B attention probe: ${DEBUG_VISUALIZE}"

cd "${GROOT_REPO}"
RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"
DEBUG_WATCHER_PID=""
DEBUG_STOP_FILE="/tmp/franka_attention_watcher_$$.stop"

stop_debug_watcher() {
    if [ -n "${DEBUG_WATCHER_PID}" ]; then
        touch "${DEBUG_STOP_FILE}"
        wait "${DEBUG_WATCHER_PID}" || true
        rm -f -- "${DEBUG_STOP_FILE}"
    fi
}
trap stop_debug_watcher EXIT

if [ "${DEBUG_VISUALIZE}" = "1" ]; then
    mkdir -p "${RUN_DIR}" "${DEBUG_OUTPUT_DIR}"
    # Dataset frames remain local until the user explicitly reviews and syncs the run.
    WANDB_MODE=offline "${VENV_PATH}/bin/python" tools/watch_franka_checkpoints.py \
        --run-dir "${RUN_DIR}" \
        --dataset "${DATASET_PATH}" \
        --output-dir "${DEBUG_OUTPUT_DIR}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-prefix "${EXPERIMENT_NAME}-debug" \
        "${DEBUG_VIS_EPISODE_ARGS[@]}" \
        --frame-step "${DEBUG_VIS_FRAME_STEP}" \
        --action-group "${DEBUG_VIS_ACTION_GROUP}" \
        --device "${DEBUG_VIS_DEVICE}" \
        --min-step "${DEBUG_VIS_MIN_STEP}" \
        --save-steps "${SAVE_STEPS}" \
        --every-n-checkpoints "${DEBUG_VIS_EVERY_N_CHECKPOINTS}" \
        --parent-pid "$$" \
        --stop-file "${DEBUG_STOP_FILE}" \
        > "${RUN_DIR}/attention-watcher.log" 2>&1 &
    DEBUG_WATCHER_PID="$!"
    echo "  attention watcher pid/log: ${DEBUG_WATCHER_PID}/${RUN_DIR}/attention-watcher.log"
    echo "  attention W&B mode: offline (manual wandb sync after review)"
fi
NUM_GPUS="${NUM_GPUS}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
USE_WANDB="${USE_WANDB}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
SHARD_SIZE="${SHARD_SIZE}" \
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH}" \
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE}" \
bash examples/finetune.sh \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --modality-config-path examples/Franka/franka_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project "${WANDB_PROJECT}" \
    --state-dropout-prob "${STATE_DROPOUT_PROB}" \
    --color-jitter-params "${COLOR_JITTER_PARAMS}" \
    --use-percentiles true \
    "${IMAGE_AUG_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
