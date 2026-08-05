#!/usr/bin/env bash
set -Eeuo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
PANEL_STATE="${PANEL_STATE:-staging}"
CAMERA_PREVIEW_PORT="${CAMERA_PREVIEW_PORT:-8081}"
CAMERA_PREVIEW_FPS="${CAMERA_PREVIEW_FPS:-10}"
echo "Scene:   http://localhost:8080 (or http://<server-ip>:8080)"
echo "Cameras: http://localhost:${CAMERA_PREVIEW_PORT} (or http://<server-ip>:${CAMERA_PREVIEW_PORT})"
docker exec -it -e DISPLAY= -e PYTHONDONTWRITEBYTECODE=1 "${CONTAINER_NAME}" bash -lc \
    "cd /workspace/isaaclab && \
     ./isaaclab.sh -p \
       /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
       --device cuda:0 --viz viser --panel-state ${PANEL_STATE} --disable-cuda-graph \
       --camera-preview --camera-preview-port ${CAMERA_PREVIEW_PORT} --camera-preview-fps ${CAMERA_PREVIEW_FPS}"
