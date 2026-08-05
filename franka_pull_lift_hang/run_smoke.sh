#!/usr/bin/env bash
set -Eeuo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
STEPS="${STEPS:-20}"
PANEL_STATE="${PANEL_STATE:-staging}"
docker exec -e DISPLAY= -e PYTHONDONTWRITEBYTECODE=1 "${CONTAINER_NAME}" bash -lc \
    "cd /workspace/isaaclab && \
     ./isaaclab.sh -p \
       /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
       --device cuda:0 --viz none --panel-state ${PANEL_STATE} --disable-cuda-graph --max-steps ${STEPS}"
