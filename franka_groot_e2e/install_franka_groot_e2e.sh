#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SCRIPTS_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
SCRIPTS_REPO="${SCRIPTS_REPO:-}"
GROOT_REPO="${GROOT_REPO:-}"
ARENA_REPO="${ARENA_REPO:-}"
MODELS_ROOT="${MODELS_ROOT:-}"
ISAAC_VENV="${ISAAC_VENV:-}"
UV_VENV="${UV_VENV:-}"
FFMPEG_RUNTIME="${FFMPEG_RUNTIME:-}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
GROOT_BRANCH="${GROOT_BRANCH:-jryu/franka-demo}"
ARENA_BRANCH="${ARENA_BRANCH:-jryu/franka-demo}"
SCRIPTS_BRANCH="${SCRIPTS_BRANCH:-main}"
SKIP_SYSTEM_PACKAGES=0
SKIP_MODEL_DOWNLOAD=0
ACCEPT_EULA_FLAG=0
PRINT_CONFIG=0

usage() {
    cat <<'EOF'
Install the Franka synthetic-data → GR00T → Arena workflow in isolated venvs.

Usage:
  install_franka_groot_e2e.sh [options]

Path options:
  --workspace-root PATH  Parent used for all unspecified paths (default: /workspace)
  --scripts-repo PATH    IsaacLab-Scripts checkout (default: this script's repo)
  --groot-repo PATH      Isaac-GR00T checkout (default: WORKSPACE_ROOT/Isaac-GR00T)
  --arena-repo PATH      IsaacLab-Arena checkout (default: WORKSPACE_ROOT/IsaacLab-Arena)
  --models-root PATH     Local model/cache parent (default: WORKSPACE_ROOT/models)
  --isaac-venv PATH      Isaac generation venv (default: WORKSPACE_ROOT/env_isaaclab)
  --uv-venv PATH         uv bootstrap venv (default: WORKSPACE_ROOT/.uv-bootstrap)
  --ffmpeg-runtime PATH  TorchCodec-compatible FFmpeg env (default: WORKSPACE_ROOT/.tools/ffmpeg-7)
  --micromamba-root PATH Micromamba bootstrap directory (default: WORKSPACE_ROOT/.tools/micromamba)
  --python PATH          Python 3.12 executable (default: /usr/bin/python3)

Repository options:
  --groot-branch NAME    Isaac-GR00T branch (default: jryu/franka-demo)
  --arena-branch NAME    IsaacLab-Arena branch (default: jryu/franka-demo)
  --scripts-branch NAME  IsaacLab-Scripts branch (default: main)

Install controls:
  --accept-eula          Accept the NVIDIA Omniverse/Isaac Sim EULA
  --skip-system-packages Do not run apt-get
  --skip-model-download  Install code and venvs without downloading model weights
  --print-config         Print resolved paths and exit without changing anything
  -h, --help             Show this help

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
        --workspace-root)
            need_value "$@"
            WORKSPACE_ROOT=$2
            shift 2
            ;;
        --scripts-repo)
            need_value "$@"
            SCRIPTS_REPO=$2
            shift 2
            ;;
        --groot-repo)
            need_value "$@"
            GROOT_REPO=$2
            shift 2
            ;;
        --arena-repo)
            need_value "$@"
            ARENA_REPO=$2
            shift 2
            ;;
        --models-root)
            need_value "$@"
            MODELS_ROOT=$2
            shift 2
            ;;
        --isaac-venv)
            need_value "$@"
            ISAAC_VENV=$2
            shift 2
            ;;
        --uv-venv)
            need_value "$@"
            UV_VENV=$2
            shift 2
            ;;
        --ffmpeg-runtime)
            need_value "$@"
            FFMPEG_RUNTIME=$2
            shift 2
            ;;
        --micromamba-root)
            need_value "$@"
            MICROMAMBA_ROOT=$2
            shift 2
            ;;
        --python)
            need_value "$@"
            PYTHON_BIN=$2
            shift 2
            ;;
        --groot-branch)
            need_value "$@"
            GROOT_BRANCH=$2
            shift 2
            ;;
        --arena-branch)
            need_value "$@"
            ARENA_BRANCH=$2
            shift 2
            ;;
        --scripts-branch)
            need_value "$@"
            SCRIPTS_BRANCH=$2
            shift 2
            ;;
        --accept-eula)
            ACCEPT_EULA_FLAG=1
            shift
            ;;
        --skip-system-packages)
            SKIP_SYSTEM_PACKAGES=1
            shift
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=1
            shift
            ;;
        --print-config)
            PRINT_CONFIG=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

WORKSPACE_ROOT="$(realpath -m -- "${WORKSPACE_ROOT}")"
SCRIPTS_REPO="$(realpath -m -- "${SCRIPTS_REPO:-${DEFAULT_SCRIPTS_REPO}}")"
GROOT_REPO="$(realpath -m -- "${GROOT_REPO:-${WORKSPACE_ROOT}/Isaac-GR00T}")"
ARENA_REPO="$(realpath -m -- "${ARENA_REPO:-${WORKSPACE_ROOT}/IsaacLab-Arena}")"
MODELS_ROOT="$(realpath -m -- "${MODELS_ROOT:-${WORKSPACE_ROOT}/models}")"
ISAAC_VENV="$(realpath -m -- "${ISAAC_VENV:-${WORKSPACE_ROOT}/env_isaaclab}")"
UV_VENV="$(realpath -m -- "${UV_VENV:-${WORKSPACE_ROOT}/.uv-bootstrap}")"
FFMPEG_RUNTIME="$(realpath -m -- "${FFMPEG_RUNTIME:-${WORKSPACE_ROOT}/.tools/ffmpeg-7}")"
MICROMAMBA_ROOT="$(realpath -m -- "${MICROMAMBA_ROOT:-${WORKSPACE_ROOT}/.tools/micromamba}")"
MICROMAMBA_BIN="${MICROMAMBA_ROOT}/bin/micromamba"
PYTHON_BIN="$(normalize_executable_path "${PYTHON_BIN}")"
BASE_MODEL_PATH="${MODELS_ROOT}/GR00T-N1.7-3B"
COSMOS_MODEL_PATH="${MODELS_ROOT}/Cosmos-Reason2-2B"
HF_HOME="${MODELS_ROOT}/huggingface-cache"
ENV_FILE="${WORKSPACE_ROOT}/franka_groot_env.sh"

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
  Isaac venv     : ${ISAAC_VENV}
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

if [ "${ACCEPT_EULA_FLAG}" != "1" ] && [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    echo "Isaac Sim installation requires EULA acceptance. Re-run with --accept-eula." >&2
    exit 2
fi
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null; then
    echo "Python 3.12 is required: ${PYTHON_BIN}" >&2
    exit 2
fi
install_system_packages() {
    if [ "${SKIP_SYSTEM_PACKAGES}" = "1" ]; then
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
if ! command -v git >/dev/null; then
    echo "git is required; install it or enable system package installation" >&2
    exit 2
fi
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

if [ ! -x "${ISAAC_VENV}/bin/python" ]; then
    "${PYTHON_BIN}" -m venv "${ISAAC_VENV}"
fi
"${ISAAC_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ISAAC_VENV}/bin/python" -m pip install \
    --extra-index-url https://pypi.nvidia.com \
    "isaaclab[isaacsim,all]==3.0.0b2.post1"
"${ISAAC_VENV}/bin/python" -m pip install msgpack-numpy==0.4.8 pyzmq==27.1.0
"${ISAAC_VENV}/bin/python" -m pip check

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

(
    cd "${ARENA_REPO}"
    "${UV_BIN}" sync --frozen --python "${PYTHON_BIN}"
)
"${UV_BIN}" pip install \
    --python "${ARENA_REPO}/.venv/bin/python" \
    msgpack-numpy==0.4.8 pyzmq==27.0.1
"${UV_BIN}" pip check --python "${ARENA_REPO}/.venv/bin/python"

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
    printf 'export PATH=%q${PATH:+:${PATH}}\n' "${FFMPEG_RUNTIME}/bin"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' "${FFMPEG_RUNTIME}/lib"
    printf 'export ISAAC_PYTHON=%q\n' "${ISAAC_VENV}/bin/python"
    printf 'export ARENA_PYTHON=%q\n' "${ISAAC_VENV}/bin/python"
    printf 'export GROOT_PYTHON=%q\n' "${GROOT_REPO}/.venv/bin/python"
    printf 'export OMNI_KIT_ACCEPT_EULA=YES\n'
    printf 'export ACCEPT_EULA=Y\n'
    printf 'export PRIVACY_CONSENT=Y\n'
} > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi is unavailable; GPU runtime must be exposed before generation/training." >&2
fi

cat <<EOF

Installation complete.
  Load paths: source ${ENV_FILE}
  W&B login : ${GROOT_REPO}/.venv/bin/wandb login
  Launch v5 : bash ${SCRIPTS_REPO}/franka_groot_e2e/run_v5_waypoint10_recovery1.sh \\
                --workspace-root ${WORKSPACE_ROOT}
EOF
