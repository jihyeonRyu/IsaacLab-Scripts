#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${SCRIPT_DIR}/dataset"
mkdir -p "${DATASET_DIR}"
chmod 0777 "${DATASET_DIR}"
CONTAINER_NAME="${CONTAINER_NAME:-franka-dual-arm-hand-hang}"
IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    existing_image="$(docker container inspect "${CONTAINER_NAME}" --format '{{.Config.Image}}')"
    if [ "${existing_image}" != "${IMAGE}" ]; then
        echo "Container ${CONTAINER_NAME} uses ${existing_image}; expected ${IMAGE}." >&2
        echo "Choose another CONTAINER_NAME or remove the old container explicitly." >&2
        exit 2
    fi
    existing_entrypoint="$(docker container inspect "${CONTAINER_NAME}" --format '{{json .Config.Entrypoint}}')"
    if [ "${existing_entrypoint}" != '["bash"]' ]; then
        echo "Container ${CONTAINER_NAME} has entrypoint ${existing_entrypoint}; expected [\"bash\"]." >&2
        echo "Recreate this dedicated container with the updated launcher." >&2
        exit 3
    fi
    docker start "${CONTAINER_NAME}" >/dev/null
    echo "Container is running: ${CONTAINER_NAME}"
    exit 0
fi

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --entrypoint bash \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 16g \
    -e ACCEPT_EULA=Y \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e PRIVACY_CONSENT=Y \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${REPO_ROOT}:/workspace/IsaacLab-Scripts:rw" \
    "${IMAGE}" -lc "sleep infinity" >/dev/null

echo "Created and started: ${CONTAINER_NAME}"
echo "Mounted: ${REPO_ROOT} -> /workspace/IsaacLab-Scripts"
echo "Dataset: ${DATASET_DIR} -> /workspace/IsaacLab-Scripts/franka_pull_lift_hang/dataset (via repository bind mount)"
