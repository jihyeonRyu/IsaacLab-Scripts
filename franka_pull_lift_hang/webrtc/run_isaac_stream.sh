#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=/workspace/IsaacLab-Scripts/franka_pull_lift_hang
PID_FILE="${PROJECT_DIR}/webrtc/runtime/isaac.pid"
LOG_FILE="${PROJECT_DIR}/webrtc/runtime/isaac.log"
PANEL_STATE="${PANEL_STATE:-staging}"
ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49102}"
ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT:-47992}"
AUTO_OPS="${AUTO_OPS:-0}"
CAMERA_PREVIEW_PORT="${CAMERA_PREVIEW_PORT:-8081}"

mkdir -p "${PROJECT_DIR}/webrtc/runtime"
cd /workspace/isaaclab
export PID_FILE PANEL_STATE PUBLIC_IP
export ISAACSIM_SIGNAL_PORT ISAACSIM_STREAM_PORT AUTO_OPS CAMERA_PREVIEW_PORT

exec setsid bash -c '
echo $$ > "$PID_FILE"
auto_ops_args=()
if [ "$AUTO_OPS" = "1" ]; then
  auto_ops_args+=(--auto-ops --max-steps 4800)
fi
exec ./isaaclab.sh -p \
  /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dual_franka_picture_hanging_scene.py \
  --device cuda:0 --viz kit --livestream 2 --panel-state "$PANEL_STATE" --disable-cuda-graph \
  --camera-preview --camera-preview-port "$CAMERA_PREVIEW_PORT" --camera-preview-fps 10 \
  "${auto_ops_args[@]}" \
  --kit_args "--/renderer/multiGpu/enabled=false \
--/rtx-transient/dlssg/enabled=false \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp=$PUBLIC_IP \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=$ISAACSIM_SIGNAL_PORT \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=$ISAACSIM_STREAM_PORT"
' >"${LOG_FILE}" 2>&1
