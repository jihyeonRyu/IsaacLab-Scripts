#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIEWER_IMAGE="${VIEWER_IMAGE:-franka-webrtc-web-viewer:1.14.2}"
ISAACSIM_HOST="${ISAACSIM_HOST:-127.0.0.1}"
ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49102}"
ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT:-47992}"

docker build \
    --build-arg "ISAACSIM_HOST=${ISAACSIM_HOST}" \
    --build-arg "ISAACSIM_SIGNAL_PORT=${ISAACSIM_SIGNAL_PORT}" \
    --build-arg "ISAACSIM_STREAM_PORT=${ISAACSIM_STREAM_PORT}" \
    --label "franka-webrtc.host=${ISAACSIM_HOST}" \
    --label "franka-webrtc.signal-port=${ISAACSIM_SIGNAL_PORT}" \
    --label "franka-webrtc.stream-port=${ISAACSIM_STREAM_PORT}" \
    -t "${VIEWER_IMAGE}" \
    "${SCRIPT_DIR}"
