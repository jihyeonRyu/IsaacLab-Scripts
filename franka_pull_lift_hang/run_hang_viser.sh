#!/usr/bin/env bash
set -Eeuo pipefail

# Canonical launcher for the Hang AutoOps debug viewer.
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
VISER_PORT="${VISER_PORT:-8080}"
CAMERA_PREVIEW_PORT="${CAMERA_PREVIEW_PORT:-8081}"
CAMERA_PREVIEW_FPS="${CAMERA_PREVIEW_FPS:-10}"
ENABLE_CAMERA_PREVIEW="${ENABLE_CAMERA_PREVIEW:-0}"
LOG_PATH="${LOG_PATH:-/tmp/franka_hang_viser.log}"

if [[ "${ENABLE_CAMERA_PREVIEW}" == "1" ]]; then
    CAMERA_ARGS="--camera-preview --camera-preview-port ${CAMERA_PREVIEW_PORT} --camera-preview-fps ${CAMERA_PREVIEW_FPS}"
else
    CAMERA_ARGS="--disable-task-cameras"
fi

docker start "${CONTAINER_NAME}" >/dev/null

# Exactly one scene process may own the Viser port.
docker exec "${CONTAINER_NAME}" bash -lc '
    pids="$(pgrep -f "[d]ual_franka_picture_hanging_scene.py" || true)"
    if [[ -n "${pids}" ]]; then
        kill -TERM ${pids} 2>/dev/null || true
        sleep 2
        remaining="$(pgrep -f "[d]ual_franka_picture_hanging_scene.py" || true)"
        [[ -z "${remaining}" ]] || kill -KILL ${remaining} 2>/dev/null || true
    fi
'

docker exec -d \
    -e DISPLAY= \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "${CONTAINER_NAME}" bash -lc \
    "cd /workspace/isaaclab && exec ./isaaclab.sh -p \
      /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
      --device cuda:0 --viz viser --panel-state staging --disable-cuda-graph \
      ${CAMERA_ARGS} \
      --auto-ops --auto-ops-task hang --hold-after-auto-ops \
      --max-steps 0 --substeps 4 \
      >${LOG_PATH} 2>&1"

echo "Hang Viser is starting (Newton + Pink)."
echo "Scene:   http://localhost:${VISER_PORT}"
if [[ "${ENABLE_CAMERA_PREVIEW}" == "1" ]]; then
    echo "Cameras: http://localhost:${CAMERA_PREVIEW_PORT}"
else
    echo "Cameras: disabled for real-time debug speed (set ENABLE_CAMERA_PREVIEW=1 to enable)"
fi
echo "Log:     docker exec ${CONTAINER_NAME} tail -f ${LOG_PATH}"
