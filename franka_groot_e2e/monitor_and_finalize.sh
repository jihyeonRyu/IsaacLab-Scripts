#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
SCRIPTS_REPO="${SCRIPTS_REPO:-${WORKSPACE_ROOT}/IsaacLab-Scripts}"
PIPELINE_PID_FILE="${PIPELINE_PID_FILE:-${WORKSPACE_ROOT}/output/franka_final_pipeline.pid}"
PIPELINE_LOG="${PIPELINE_LOG:-${WORKSPACE_ROOT}/output/franka_final_pipeline.log}"
STATE_DIR="${STATE_DIR:-${WORKSPACE_ROOT}/output/franka_e2e_pipeline_final}"
MONITOR_LOG="${MONITOR_LOG:-${WORKSPACE_ROOT}/output/franka_final_monitor.log}"
RAW_DATASET="${RAW_DATASET:-${WORKSPACE_ROOT}/output/franka_max2_robust_2000eps_seed91007}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [ ! -f "${PIPELINE_PID_FILE}" ]; then
    echo "Missing pipeline PID file: ${PIPELINE_PID_FILE}" >&2
    exit 2
fi
PIPELINE_PID="$(cat "${PIPELINE_PID_FILE}")"
if ! [[ "${PIPELINE_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid pipeline PID: ${PIPELINE_PID}" >&2
    exit 2
fi

mkdir -p "$(dirname -- "${MONITOR_LOG}")"
exec >>"${MONITOR_LOG}" 2>&1

status() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

last_stage=""
while kill -0 "${PIPELINE_PID}" 2>/dev/null; do
    stage="generation"
    for candidate in analysis conversion training_plan sft ema_attention arena_eval cleanup complete; do
        if [ -f "${STATE_DIR}/${candidate}.done" ]; then
            stage="${candidate}"
        fi
    done
    if [ "${stage}" != "${last_stage}" ]; then
        status "stage=${stage}"
        last_stage="${stage}"
    fi
    if [ "${stage}" = "generation" ] && [ -d "${RAW_DATASET}" ]; then
        episode_dirs="$(find "${RAW_DATASET}" -maxdepth 1 -type d -name 'episode_*' | wc -l)"
        result_files="$(find "${RAW_DATASET}" -maxdepth 3 -type f -path '*/logs/result.json' | wc -l)"
        status "generation episode_dirs=${episode_dirs}/2000 closed=${result_files}/2000"
    fi
    if rg -q 'FAILED stage=' "${STATE_DIR}/status.log" 2>/dev/null; then
        status "pipeline reported failure"
        tail -n 100 "${PIPELINE_LOG}"
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

if [ ! -f "${STATE_DIR}/complete.done" ]; then
    status "pipeline PID ${PIPELINE_PID} exited before complete.done"
    tail -n 200 "${PIPELINE_LOG}"
    exit 1
fi

status "pipeline complete; packaging final assets and documentation"
python3 \
    "${SCRIPTS_REPO}/franka_groot_e2e/scripts/05_finalize/finalize_franka_run.py" \
    --workspace-root "${WORKSPACE_ROOT}" \
    --scripts-repo "${SCRIPTS_REPO}" \
    --pipeline-log "${PIPELINE_LOG}"

status "committing final evidence to IsaacLab-Scripts"
git -C "${SCRIPTS_REPO}" add \
    franka_groot_e2e/README.md \
    franka_groot_e2e/assets
if git -C "${SCRIPTS_REPO}" diff --cached --quiet; then
    status "no repository changes produced by finalization"
else
    git -C "${SCRIPTS_REPO}" commit -m "Publish latest Franka GR00T results"
    git -C "${SCRIPTS_REPO}" push origin main
fi
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/published.done"
status "final results published"
