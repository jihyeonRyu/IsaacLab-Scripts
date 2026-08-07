#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/webrtc/runtime"
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
VIEWER_CONTAINER="${VIEWER_CONTAINER:-franka-webrtc-web-viewer}"
VIEWER_IMAGE="${VIEWER_IMAGE:-franka-webrtc-web-viewer:1.14.2}"
WEB_VIEWER_PORT="${WEB_VIEWER_PORT:-8002}"
ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49102}"
ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT:-47992}"
PANEL_STATE="${PANEL_STATE:-staging}"

if [ -z "${ISAACSIM_HOST:-}" ]; then
    ISAACSIM_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
    ISAACSIM_HOST="${ISAACSIM_HOST:-127.0.0.1}"
fi

mkdir -p "${RUNTIME_DIR}"
chmod 0777 "${RUNTIME_DIR}"
bash "${SCRIPT_DIR}/docker_start.sh"

EXPECTED_HOST="$(docker image inspect "${VIEWER_IMAGE}" --format '{{index .Config.Labels "franka-webrtc.host"}}' 2>/dev/null || true)"
EXPECTED_SIGNAL_PORT="$(docker image inspect "${VIEWER_IMAGE}" --format '{{index .Config.Labels "franka-webrtc.signal-port"}}' 2>/dev/null || true)"
EXPECTED_STREAM_PORT="$(docker image inspect "${VIEWER_IMAGE}" --format '{{index .Config.Labels "franka-webrtc.stream-port"}}' 2>/dev/null || true)"
if [ "${EXPECTED_HOST}" != "${ISAACSIM_HOST}" ] \
    || [ "${EXPECTED_SIGNAL_PORT}" != "${ISAACSIM_SIGNAL_PORT}" ] \
    || [ "${EXPECTED_STREAM_PORT}" != "${ISAACSIM_STREAM_PORT}" ]; then
    ISAACSIM_HOST="${ISAACSIM_HOST}" \
    ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT}" \
    ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT}" \
        bash "${SCRIPT_DIR}/webrtc/build_viewer.sh"
fi

if docker container inspect "${VIEWER_CONTAINER}" >/dev/null 2>&1; then
    docker rm -f "${VIEWER_CONTAINER}" >/dev/null
fi

docker run -d --name "${VIEWER_CONTAINER}" --network host --no-healthcheck \
    "${VIEWER_IMAGE}" npx vite preview --host 0.0.0.0 --port "${WEB_VIEWER_PORT}" \
    >"${RUNTIME_DIR}/viewer.container-id"

docker exec "${CONTAINER_NAME}" bash -lc '
pid_file=/workspace/IsaacLab-Scripts/franka_pull_lift_hang/webrtc/runtime/isaac.pid
if [ -s "${pid_file}" ]; then
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
        kill -TERM -- "-${old_pid}" 2>/dev/null || kill -TERM "${old_pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
fi'

docker exec -d \
    -e "PUBLIC_IP=${ISAACSIM_HOST}" \
    -e "PANEL_STATE=${PANEL_STATE}" \
    -e "AUTO_OPS=${AUTO_OPS:-0}" \
    -e "CAMERA_PREVIEW_PORT=${CAMERA_PREVIEW_PORT:-8081}" \
    -e "ISAACSIM_SIGNAL_PORT=${ISAACSIM_SIGNAL_PORT}" \
    -e "ISAACSIM_STREAM_PORT=${ISAACSIM_STREAM_PORT}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "${CONTAINER_NAME}" \
    bash /workspace/IsaacLab-Scripts/franka_pull_lift_hang/webrtc/run_isaac_stream.sh

echo "Isaac Sim is starting with the dual-Franka picture-hanging scene."
echo "Viewer: http://${ISAACSIM_HOST}:${WEB_VIEWER_PORT}"
echo "Log: ${RUNTIME_DIR}/isaac.log"
echo "Only one WebRTC client can connect at a time."
