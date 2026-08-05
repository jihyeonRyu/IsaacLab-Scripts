#!/usr/bin/env bash
set -Eeuo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
PANEL_STATE="${PANEL_STATE:-staging}"
echo "Starting native Newton viewer on DISPLAY=${DISPLAY:-:0}."
docker exec -it \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "${CONTAINER_NAME}" bash -lc \
    "cd /workspace/isaaclab && \
     ./isaaclab.sh -p \
       /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
       --device cuda:0 --viz newton --panel-state ${PANEL_STATE} --disable-cuda-graph"
