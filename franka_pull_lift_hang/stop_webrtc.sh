#!/usr/bin/env bash
set -Eeuo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
VIEWER_CONTAINER="${VIEWER_CONTAINER:-franka-webrtc-web-viewer}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker exec "${CONTAINER_NAME}" bash -lc '
pid_file=/workspace/IsaacLab-Scripts/franka_pull_lift_hang/webrtc/runtime/isaac.pid
if [ -s "${pid_file}" ]; then
    pid="$(cat "${pid_file}")"
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
fi'
fi

if docker container inspect "${VIEWER_CONTAINER}" >/dev/null 2>&1; then
    docker rm -f "${VIEWER_CONTAINER}" >/dev/null
fi

echo "Stopped the Franka WebRTC session."
