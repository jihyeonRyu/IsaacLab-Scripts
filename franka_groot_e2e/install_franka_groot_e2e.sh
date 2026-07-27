#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCRIPTS_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REQUIRED_BASE_IMAGE="nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1"
SCRIPTS_REPO="${SCRIPTS_REPO:-}"
GROOT_REPO="${GROOT_REPO:-}"
ARENA_REPO="${ARENA_REPO:-}"
MODELS_ROOT="${MODELS_ROOT:-}"
ISAAC_VENV="${ISAAC_VENV:-}"
ISAAC_MODE="${ISAAC_MODE:-ngc-container}"
BASE_ISAAC_PYTHON="${BASE_ISAAC_PYTHON:-/isaac-sim/python.sh}"
BASE_ISAAC_LAB_ROOT="${BASE_ISAAC_LAB_ROOT:-/workspace/isaaclab}"
UV_VENV="${UV_VENV:-}"
FFMPEG_RUNTIME="${FFMPEG_RUNTIME:-}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
NUM_GPUS="${NUM_GPUS:-auto}"
GPU_IDS="${GPU_IDS:-}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
SCRIPTS_BRANCH="${SCRIPTS_BRANCH:-main}"
SKIP_SYSTEM_PACKAGES=0
SKIP_MODEL_DOWNLOAD=0
ACCEPT_EULA_FLAG=0
PRINT_CONFIG=0

usage() {
    cat <<'EOF'
Install the Franka synthetic-data → GR00T → Arena workflow inside the pinned
NVIDIA Isaac Lab NGC container. GR00T uses an isolated venv; Arena runtime
dependencies use a workspace-local user site attached to bundled Isaac Python.

Usage:
  install_franka_groot_e2e.sh [options]

Path options:
  --workspace-root PATH  Parent used for all unspecified paths (default: /workspace)
  --scripts-repo PATH    IsaacLab-Scripts checkout (default: this script's repo)
  --groot-repo PATH      Isaac-GR00T checkout (default: WORKSPACE_ROOT/Isaac-GR00T)
  --arena-repo PATH      IsaacLab-Arena checkout (default: WORKSPACE_ROOT/IsaacLab-Arena)
  --models-root PATH     Local model/cache parent (default: WORKSPACE_ROOT/models)
  --isaac-venv PATH      Isaac venv used only with --isaac-mode pip
  --base-isaac-python PATH
                         Bundled Isaac Python (default: /isaac-sim/python.sh)
  --base-isaac-lab-root PATH
                         Bundled Isaac Lab checkout (default: /workspace/isaaclab)
  --uv-venv PATH         uv bootstrap venv (default: WORKSPACE_ROOT/.uv-bootstrap)
  --ffmpeg-runtime PATH  TorchCodec-compatible FFmpeg env (default: WORKSPACE_ROOT/.tools/ffmpeg-7)
  --micromamba-root PATH Micromamba bootstrap directory (default: WORKSPACE_ROOT/.tools/micromamba)
  --python PATH          Python 3.12 bootstrap executable (default: bundled Python)

Repository options:
  --groot-branch NAME    Isaac-GR00T branch (default: jryu/franka-demo)
  --arena-branch NAME    IsaacLab-Arena branch (default: jryu/franka-demo)
  --scripts-branch NAME  IsaacLab-Scripts branch (default: main)

Compute options:
  --num-gpus auto|N      CUDA-visible GPUs recorded for the workflow (default: auto)
  --gpu-ids CSV          Physical GPU IDs (default: first NUM_GPUS visible IDs)

Install controls:
  --isaac-mode MODE      ngc-container (default) or pip
  --accept-eula          Accept the NVIDIA Omniverse/Isaac Sim EULA
  --skip-system-packages Do not run apt-get in the explicit pip fallback
  --skip-model-download  Install code and venvs without downloading model weights
  --print-config         Validate and print the resolved configuration, then exit
  -h, --help             Show this help

Required base image:
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1

The default mode refuses to run outside that container layout. Use
--isaac-mode pip only for an explicitly unsupported workstation/VM fallback.
The model repositories are public. No Hugging Face token is required.
W&B authentication is intentionally not stored by this script; run `wandb login`
once in the GR00T venv before online training.
EOF
}

need_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

normalize_executable_path() {
    local executable_path=$1
    local executable_dir
    executable_dir="$(realpath -m -- "$(dirname -- "${executable_path}")")"
    printf '%s/%s\n' "${executable_dir}" "$(basename -- "${executable_path}")"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workspace-root) need_value "$@"; WORKSPACE_ROOT=$2; shift 2 ;;
        --scripts-repo) need_value "$@"; SCRIPTS_REPO=$2; shift 2 ;;
        --groot-repo) need_value "$@"; GROOT_REPO=$2; shift 2 ;;
        --arena-repo) need_value "$@"; ARENA_REPO=$2; shift 2 ;;
        --models-root) need_value "$@"; MODELS_ROOT=$2; shift 2 ;;
        --isaac-venv) need_value "$@"; ISAAC_VENV=$2; shift 2 ;;
        --base-isaac-python) need_value "$@"; BASE_ISAAC_PYTHON=$2; shift 2 ;;
        --base-isaac-lab-root) need_value "$@"; BASE_ISAAC_LAB_ROOT=$2; shift 2 ;;
        --uv-venv) need_value "$@"; UV_VENV=$2; shift 2 ;;
        --ffmpeg-runtime) need_value "$@"; FFMPEG_RUNTIME=$2; shift 2 ;;
        --micromamba-root) need_value "$@"; MICROMAMBA_ROOT=$2; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
        --num-gpus) need_value "$@"; NUM_GPUS=$2; shift 2 ;;
        --gpu-ids) need_value "$@"; GPU_IDS=$2; shift 2 ;;
        --groot-branch) need_value "$@"; GROOT_BRANCH=$2; shift 2 ;;
        --arena-branch) need_value "$@"; ARENA_BRANCH=$2; shift 2 ;;
        --scripts-branch) need_value "$@"; SCRIPTS_BRANCH=$2; shift 2 ;;
        --isaac-mode) need_value "$@"; ISAAC_MODE=$2; shift 2 ;;
        --accept-eula) ACCEPT_EULA_FLAG=1; shift ;;
        --skip-system-packages) SKIP_SYSTEM_PACKAGES=1; shift ;;
        --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${ISAAC_MODE}" in
    ngc-container|pip) ;;
    *)
        echo "--isaac-mode must be ngc-container or pip, got ${ISAAC_MODE}" >&2
        exit 2
        ;;
esac

WORKSPACE_ROOT="$(realpath -m -- "${WORKSPACE_ROOT}")"
SCRIPTS_REPO="$(realpath -m -- "${SCRIPTS_REPO:-${DEFAULT_SCRIPTS_REPO}}")"
GROOT_REPO="$(realpath -m -- "${GROOT_REPO:-${WORKSPACE_ROOT}/Isaac-GR00T}")"
ARENA_REPO="$(realpath -m -- "${ARENA_REPO:-${WORKSPACE_ROOT}/IsaacLab-Arena}")"
MODELS_ROOT="$(realpath -m -- "${MODELS_ROOT:-${WORKSPACE_ROOT}/models}")"
ISAAC_VENV="$(realpath -m -- "${ISAAC_VENV:-${WORKSPACE_ROOT}/env_isaaclab}")"
BASE_ISAAC_PYTHON="$(normalize_executable_path "${BASE_ISAAC_PYTHON}")"
BASE_ISAAC_LAB_ROOT="$(realpath -m -- "${BASE_ISAAC_LAB_ROOT}")"
UV_VENV="$(realpath -m -- "${UV_VENV:-${WORKSPACE_ROOT}/.uv-bootstrap}")"
FFMPEG_RUNTIME="$(realpath -m -- "${FFMPEG_RUNTIME:-${WORKSPACE_ROOT}/.tools/ffmpeg-7}")"
MICROMAMBA_ROOT="$(realpath -m -- "${MICROMAMBA_ROOT:-${WORKSPACE_ROOT}/.tools/micromamba}")"
MICROMAMBA_BIN="${MICROMAMBA_ROOT}/bin/micromamba"
if [ -z "${PYTHON_BIN}" ]; then
    if [ "${ISAAC_MODE}" = "ngc-container" ]; then
        PYTHON_BIN="/isaac-sim/kit/python/bin/python3"
    else
        PYTHON_BIN="/usr/bin/python3"
    fi
fi
PYTHON_BIN="$(normalize_executable_path "${PYTHON_BIN}")"
BASE_MODEL_PATH="${MODELS_ROOT}/GR00T-N1.7-3B"
COSMOS_MODEL_PATH="${MODELS_ROOT}/Cosmos-Reason2-2B"
HF_HOME="${MODELS_ROOT}/huggingface-cache"
ENV_FILE="${WORKSPACE_ROOT}/franka_groot_env.sh"
ISAAC_PYTHONUSERBASE="${WORKSPACE_ROOT}/.isaac-python-user"

if [ "${ACCEPT_EULA_FLAG}" != "1" ] && [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    echo "Isaac Sim requires EULA acceptance. Re-run with --accept-eula." >&2
    exit 2
fi
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

if [ "${ISAAC_MODE}" = "ngc-container" ] && \
    [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    echo "Run this installer inside ${REQUIRED_BASE_IMAGE}." >&2
    echo "No Docker/OCI container marker was found." >&2
    echo "For an unsupported pip fallback, pass --isaac-mode pip explicitly." >&2
    exit 2
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null; then
    echo "Python 3.12 is required: ${PYTHON_BIN}" >&2
    exit 2
fi

if [ "${ISAAC_MODE}" = "ngc-container" ]; then
    if [ ! -x "${BASE_ISAAC_PYTHON}" ]; then
        echo "Bundled Isaac Python is missing: ${BASE_ISAAC_PYTHON}" >&2
        echo "Expected base image: ${REQUIRED_BASE_IMAGE}" >&2
        exit 2
    fi
    if [ ! -x "${BASE_ISAAC_LAB_ROOT}/isaaclab.sh" ]; then
        echo "Bundled Isaac Lab is missing: ${BASE_ISAAC_LAB_ROOT}/isaaclab.sh" >&2
        echo "Expected base image: ${REQUIRED_BASE_IMAGE}" >&2
        exit 2
    fi
    if [ ! -f /isaac-sim/VERSION ] || ! grep -q '^6\.0\.1' /isaac-sim/VERSION; then
        echo "Expected Isaac Sim 6.0.1 in ${REQUIRED_BASE_IMAGE}." >&2
        exit 2
    fi
    if [ ! -f "${BASE_ISAAC_LAB_ROOT}/VERSION" ] ||
        ! grep -q '^3\.0\.0' "${BASE_ISAAC_LAB_ROOT}/VERSION"; then
        echo "Expected Isaac Lab 3.0.0 in ${REQUIRED_BASE_IMAGE}." >&2
        exit 2
    fi
    ISAAC_PYTHON="${BASE_ISAAC_PYTHON}"
else
    ISAAC_PYTHON="${ISAAC_VENV}/bin/python"
fi

mapfile -t PHYSICAL_GPU_IDS < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
        sed '/^[[:space:]]*$/d; s/[[:space:]]//g'
)
if [ "${#PHYSICAL_GPU_IDS[@]}" -eq 0 ]; then
    echo "No NVIDIA GPU was detected. Check NVIDIA Container Toolkit and --gpus." >&2
    exit 2
fi

CUDA_VISIBLE_ID_ARRAY=()
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=',' read -r -a CUDA_VISIBLE_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
fi
if [ "${NUM_GPUS}" = "auto" ]; then
    if [ "${#CUDA_VISIBLE_ID_ARRAY[@]}" -gt 0 ]; then
        NUM_GPUS="${#CUDA_VISIBLE_ID_ARRAY[@]}"
    else
        NUM_GPUS="${#PHYSICAL_GPU_IDS[@]}"
    fi
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-gpus must be auto or a positive integer, got ${NUM_GPUS}" >&2
    exit 2
fi
if [ "${NUM_GPUS}" -gt "${#PHYSICAL_GPU_IDS[@]}" ]; then
    echo "Requested ${NUM_GPUS} GPUs, but nvidia-smi exposes only ${#PHYSICAL_GPU_IDS[@]}." >&2
    exit 2
fi

GPU_ID_ARRAY=()
if [ -n "${GPU_IDS}" ]; then
    IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
elif [ "${#CUDA_VISIBLE_ID_ARRAY[@]}" -ge "${NUM_GPUS}" ]; then
    GPU_ID_ARRAY=("${CUDA_VISIBLE_ID_ARRAY[@]:0:NUM_GPUS}")
else
    GPU_ID_ARRAY=("${PHYSICAL_GPU_IDS[@]:0:NUM_GPUS}")
fi
if [ "${#GPU_ID_ARRAY[@]}" -ne "${NUM_GPUS}" ]; then
    echo "--gpu-ids must contain exactly ${NUM_GPUS} comma-separated physical IDs." >&2
    exit 2
fi
declare -A AVAILABLE_GPU_IDS=()
declare -A SEEN_GPU_IDS=()
for gpu_id in "${PHYSICAL_GPU_IDS[@]}"; do
    AVAILABLE_GPU_IDS["${gpu_id}"]=1
done
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid physical GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    if [ -z "${AVAILABLE_GPU_IDS[${gpu_id}]:-}" ]; then
        echo "GPU ID ${gpu_id} is not visible to nvidia-smi." >&2
        exit 2
    fi
    if [ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]; then
        echo "Duplicate physical GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    SEEN_GPU_IDS["${gpu_id}"]=1
done
GPU_IDS="$(IFS=','; echo "${GPU_ID_ARRAY[*]}")"

print_config() {
    cat <<EOF
Resolved Franka GR00T installation
  workspace root : ${WORKSPACE_ROOT}
  scripts repo   : ${SCRIPTS_REPO} (${SCRIPTS_BRANCH})
  GR00T repo     : ${GROOT_REPO} (${GROOT_BRANCH})
  Arena repo     : ${ARENA_REPO} (${ARENA_BRANCH})
  models root    : ${MODELS_ROOT}
  GR00T model    : ${BASE_MODEL_PATH}
  Cosmos model   : ${COSMOS_MODEL_PATH}
  HF cache       : ${HF_HOME}
  required image : ${REQUIRED_BASE_IMAGE}
  Isaac mode     : ${ISAAC_MODE}
  Isaac Python   : ${ISAAC_PYTHON}
  Isaac Lab root : ${BASE_ISAAC_LAB_ROOT}
  Isaac user site: ${ISAAC_PYTHONUSERBASE}
  pip fallback   : ${ISAAC_VENV}
  GPUs / IDs     : ${NUM_GPUS} / ${GPU_IDS}
  uv venv        : ${UV_VENV}
  FFmpeg runtime : ${FFMPEG_RUNTIME}
  micromamba     : ${MICROMAMBA_BIN}
  Python         : ${PYTHON_BIN}
  env file       : ${ENV_FILE}
EOF
}

print_config
if [ "${PRINT_CONFIG}" = "1" ]; then
    exit 0
fi

install_system_packages() {
    if [ "${ISAAC_MODE}" = "ngc-container" ] || [ "${SKIP_SYSTEM_PACKAGES}" = "1" ]; then
        return
    fi
    local apt_prefix=()
    if [ "$(id -u)" -ne 0 ]; then
        if ! command -v sudo >/dev/null; then
            echo "System package installation needs root or sudo; use --skip-system-packages if already prepared." >&2
            exit 2
        fi
        apt_prefix=(sudo)
    fi
    "${apt_prefix[@]}" apt-get update
    "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates curl git bzip2 build-essential cmake ninja-build pkg-config \
        python3.12-venv python3-pip \
        libice6 libsm6 libxt6t64 libgl1 libegl1 libglib2.0-0 \
        libx11-6 libxrender1 libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1
}

install_ffmpeg_runtime() {
    if [ ! -x "${FFMPEG_RUNTIME}/bin/ffmpeg" ]; then
        if [ ! -x "${MICROMAMBA_BIN}" ]; then
            if ! command -v curl >/dev/null || ! command -v tar >/dev/null; then
                echo "curl and tar are required to bootstrap the FFmpeg runtime." >&2
                exit 2
            fi
            mkdir -p -- "${MICROMAMBA_ROOT}"
            curl -LsSf https://micro.mamba.pm/api/micromamba/linux-64/latest |
                tar -xj -C "${MICROMAMBA_ROOT}" bin/micromamba
        fi
        MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT}/root" \
            "${MICROMAMBA_BIN}" create \
            -p "${FFMPEG_RUNTIME}" \
            -c conda-forge \
            "ffmpeg>=4,<8" \
            -y
    fi

    local ffmpeg_version
    local ffmpeg_major
    ffmpeg_version="$("${FFMPEG_RUNTIME}/bin/ffmpeg" -version | awk 'NR == 1 { print $3 }')"
    ffmpeg_major="${ffmpeg_version%%.*}"
    if ! [[ "${ffmpeg_major}" =~ ^[4-7]$ ]]; then
        echo "TorchCodec 0.8 requires FFmpeg 4-7; found ${ffmpeg_version} in ${FFMPEG_RUNTIME}." >&2
        exit 2
    fi
    export PATH="${FFMPEG_RUNTIME}/bin${PATH:+:${PATH}}"
    export LD_LIBRARY_PATH="${FFMPEG_RUNTIME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

clone_or_validate() {
    local url=$1
    local path=$2
    local branch=$3
    if [ -e "${path}" ] && [ ! -d "${path}/.git" ]; then
        if [ -d "${path}" ] && [ -z "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            rmdir -- "${path}"
        else
            echo "Refusing to overwrite a non-Git path: ${path}" >&2
            exit 2
        fi
    fi
    if [ ! -d "${path}/.git" ]; then
        mkdir -p -- "$(dirname -- "${path}")"
        git clone --branch "${branch}" --single-branch "${url}" "${path}"
    fi
    local current_branch
    current_branch="$(git -C "${path}" branch --show-current)"
    if [ "${current_branch}" != "${branch}" ]; then
        echo "Expected ${path} on branch ${branch}; found ${current_branch:-detached HEAD}" >&2
        exit 2
    fi
}

install_system_packages
for command_name in git curl tar; do
    if ! command -v "${command_name}" >/dev/null; then
        echo "Required command is missing: ${command_name}" >&2
        exit 2
    fi
done
mkdir -p -- "${WORKSPACE_ROOT}" "${MODELS_ROOT}" "${HF_HOME}"
install_ffmpeg_runtime

clone_or_validate \
    https://github.com/jihyeonRyu/IsaacLab-Scripts.git \
    "${SCRIPTS_REPO}" \
    "${SCRIPTS_BRANCH}"
clone_or_validate \
    https://github.com/jihyeonRyu/Isaac-GR00T.git \
    "${GROOT_REPO}" \
    "${GROOT_BRANCH}"
clone_or_validate \
    https://github.com/jihyeonRyu/IsaacLab-Arena.git \
    "${ARENA_REPO}" \
    "${ARENA_BRANCH}"

if [ "${ISAAC_MODE}" = "ngc-container" ]; then
    mkdir -p -- "${ISAAC_PYTHONUSERBASE}"
    PYTHONUSERBASE="${ISAAC_PYTHONUSERBASE}" "${ISAAC_PYTHON}" -m pip install \
        --user \
        msgpack-numpy==0.4.8 \
        pyzmq==27.1.0
    PYTHONUSERBASE="${ISAAC_PYTHONUSERBASE}" "${ISAAC_PYTHON}" -m pip install \
        --user \
        -e "${ARENA_REPO}"
    PYTHONUSERBASE="${ISAAC_PYTHONUSERBASE}" "${ISAAC_PYTHON}" -m pip check
    PYTHONUSERBASE="${ISAAC_PYTHONUSERBASE}" "${ISAAC_PYTHON}" -c \
        'import isaaclab, isaaclab_arena, msgpack_numpy, zmq; print("Bundled Isaac + Arena runtime: OK")'
else
    if [ ! -x "${ISAAC_VENV}/bin/python" ]; then
        "${PYTHON_BIN}" -m venv "${ISAAC_VENV}"
    fi
    "${ISAAC_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
    "${ISAAC_VENV}/bin/python" -m pip install \
        --extra-index-url https://pypi.nvidia.com \
        "isaaclab[isaacsim,all]==3.0.0b2.post1"
    "${ISAAC_VENV}/bin/python" -m pip install msgpack-numpy==0.4.8 pyzmq==27.1.0
    "${ISAAC_VENV}/bin/python" -m pip install -e "${ARENA_REPO}"
    "${ISAAC_VENV}/bin/python" -m pip check
fi

if [ ! -x "${UV_VENV}/bin/python" ]; then
    "${PYTHON_BIN}" -m venv "${UV_VENV}"
fi
"${UV_VENV}/bin/python" -m pip install --upgrade pip
"${UV_VENV}/bin/python" -m pip install uv==0.11.31
UV_BIN="${UV_VENV}/bin/uv"

(
    cd "${GROOT_REPO}"
    "${UV_BIN}" sync --frozen --python "${PYTHON_BIN}"
)
"${UV_BIN}" pip check --python "${GROOT_REPO}/.venv/bin/python"
"${GROOT_REPO}/.venv/bin/python" -c \
    'from torchcodec.decoders import VideoDecoder; print("TorchCodec FFmpeg runtime: OK")'

if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ]; then
    HF_HOME="${HF_HOME}" "${GROOT_REPO}/.venv/bin/hf" download \
        nvidia/GR00T-N1.7-3B \
        --local-dir "${BASE_MODEL_PATH}"
    HF_HOME="${HF_HOME}" "${GROOT_REPO}/.venv/bin/hf" download \
        nvidia/Cosmos-Reason2-2B \
        --local-dir "${COSMOS_MODEL_PATH}"
fi

{
    printf '# Generated by %q\n' "$0"
    printf 'export WORKSPACE_ROOT=%q\n' "${WORKSPACE_ROOT}"
    printf 'export SCRIPTS_REPO=%q\n' "${SCRIPTS_REPO}"
    printf 'export GROOT_REPO=%q\n' "${GROOT_REPO}"
    printf 'export ARENA_REPO=%q\n' "${ARENA_REPO}"
    printf 'export MODELS_ROOT=%q\n' "${MODELS_ROOT}"
    printf 'export BASE_MODEL_PATH=%q\n' "${BASE_MODEL_PATH}"
    printf 'export GROOT_COSMOS_MODEL_PATH=%q\n' "${COSMOS_MODEL_PATH}"
    printf 'export HF_HOME=%q\n' "${HF_HOME}"
    printf 'export FFMPEG_RUNTIME=%q\n' "${FFMPEG_RUNTIME}"
    printf 'export REQUIRED_BASE_IMAGE=%q\n' "${REQUIRED_BASE_IMAGE}"
    printf 'export ISAAC_MODE=%q\n' "${ISAAC_MODE}"
    printf 'export PYTHONUSERBASE=%q\n' "${ISAAC_PYTHONUSERBASE}"
    printf 'export PATH=%q${PATH:+:${PATH}}\n' "${FFMPEG_RUNTIME}/bin"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' "${FFMPEG_RUNTIME}/lib"
    printf 'export ISAAC_PATH=%q\n' /isaac-sim
    printf 'export ISAAC_LAB_ROOT=%q\n' "${BASE_ISAAC_LAB_ROOT}"
    printf 'export ISAAC_PYTHON=%q\n' "${ISAAC_PYTHON}"
    printf 'export ARENA_PYTHON=%q\n' "${ISAAC_PYTHON}"
    printf 'export GROOT_PYTHON=%q\n' "${GROOT_REPO}/.venv/bin/python"
    printf 'export NUM_GPUS=%q\n' "${NUM_GPUS}"
    printf 'export GPU_IDS=%q\n' "${GPU_IDS}"
    printf 'export OMNI_KIT_ACCEPT_EULA=YES\n'
    printf 'export ACCEPT_EULA=Y\n'
    printf 'export PRIVACY_CONSENT=Y\n'
} > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

cat <<EOF

Installation complete.
  Base image: ${REQUIRED_BASE_IMAGE}
  Isaac mode: ${ISAAC_MODE}; bundled Isaac Sim/Lab was not reinstalled
  GPUs      : ${NUM_GPUS} (${GPU_IDS})
  Load paths: source ${ENV_FILE}
  W&B login : ${GROOT_REPO}/.venv/bin/wandb login
  Launch    : bash ${SCRIPTS_REPO}/franka_groot_e2e/run_pipeline.sh \\
                --workspace-root ${WORKSPACE_ROOT} \\
                --num-gpus ${NUM_GPUS} \\
                --gpu-ids ${GPU_IDS}
EOF
