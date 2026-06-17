#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${LOOM_BENCH_MODE:-${1:-smoke}}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  smoke)
    HARNESS_ARGS=(--smoke-only)
    ;;
  tuning)
    HARNESS_ARGS=(--all --benchmark-set tuning "$@")
    ;;
  held-out)
    HARNESS_ARGS=(--all --benchmark-set held-out "$@")
    ;;
  frontend)
    HARNESS_ARGS=(--frontend-comparison --frontend-runtime "$@")
    ;;
  partial)
    if [[ "$#" -eq 0 ]]; then
      echo "partial mode requires harness arguments" >&2
      exit 2
    fi
    HARNESS_ARGS=("$@")
    ;;
  full)
    echo "full mode has been replaced by tuning, held-out, and frontend jobs" >&2
    exit 2
    ;;
  publish)
    echo "publish mode is selected at the workflow level; ci-benchmark runs one job mode at a time" >&2
    exit 2
    ;;
  args)
    HARNESS_ARGS=("$@")
    ;;
  *)
    echo "unknown benchmark mode: ${MODE}" >&2
    exit 2
    ;;
esac

IMAGE="$("${SCRIPT_DIR}/build-image.sh")"

GIT_SHA="$(git rev-parse --short=12 HEAD)"
RUN_ID="${GITEA_RUN_ID:-${GITHUB_RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}}"
ARTIFACT_ROOT="${LOOM_BENCH_ARTIFACT_ROOT:-experiments/artifacts/dgx/${GIT_SHA}/${RUN_ID}}"
LOG_DIR="${ARTIFACT_ROOT}/logs"
RESULTS_DIR="${ARTIFACT_ROOT}/results"
WORK_DIR="${ARTIFACT_ROOT}/work"

mkdir -p .container-cache "${LOG_DIR}" "${RESULTS_DIR}" "${WORK_DIR}"

cleanup_workspace_permissions() {
  docker run --rm \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc 'chown -R "${HOST_UID}:${HOST_GID}" /workspace/_build /workspace/.container-cache "/workspace/${ARTIFACT_ROOT}" 2>/dev/null || true; find /workspace -type d -name __pycache__ -prune -exec chown -R "${HOST_UID}:${HOST_GID}" {} + 2>/dev/null || true' \
    >/dev/null 2>&1 || true
}
trap cleanup_workspace_permissions EXIT

run_with_heartbeat() {
  local log_path="$1"
  shift
  local heartbeat_seconds="${LOOM_BENCH_HEARTBEAT_SECONDS:-120}"
  local started_at
  local status
  started_at="$(date -u +%s)"

  : > "${log_path}"
  "$@" > >(tee -a "${log_path}") 2>&1 &
  local cmd_pid=$!

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    sleep "${heartbeat_seconds}"
    if kill -0 "${cmd_pid}" 2>/dev/null; then
      local now elapsed
      now="$(date -u +%s)"
      elapsed=$((now - started_at))
      printf '[ci-heartbeat] %s still running after %ss: ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${elapsed}" | tee -a "${log_path}"
      printf '%q ' "$@" | tee -a "${log_path}"
      printf '\n' | tee -a "${log_path}"
    fi
  done

  set +e
  wait "${cmd_pid}"
  status=$?
  set -e
  return "${status}"
}

{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "mode=${MODE}"
  printf 'harness_args='
  printf '%q ' "${HARNESS_ARGS[@]}"
  echo
  echo "artifact_root=${ARTIFACT_ROOT}"
  echo "image=${IMAGE}"
} > "${LOG_DIR}/ci-benchmark.env"

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e UV_CACHE_DIR=/workspace/.container-cache/uv \
  -e TRITON_CACHE_DIR=/workspace/.container-cache/triton \
  -e UV_PROJECT_ENVIRONMENT=/opt/loom-venv \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "opam exec -- dune build src/loom_cli/main.exe && uv run --no-sync python -m unittest discover -s test/python -p 'test_*.py'" \
  2>&1 | tee "${LOG_DIR}/python-tests.log"

HARNESS_CMD=(
  "${SCRIPT_DIR}/run-container.sh"
  --image "${IMAGE}"
  --results-dir "${RESULTS_DIR}"
  --work-dir "${WORK_DIR}"
  -- "${HARNESS_ARGS[@]}"
)
run_with_heartbeat "${LOG_DIR}/harness.log" "${HARNESS_CMD[@]}"

print_result_csv() {
  local label="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    echo "::group::${label}"
    sed -n '1,160p' "${path}"
    echo "::endgroup::"
  fi
}

SUMMARY_DIR="${RESULTS_DIR}/runtime/summaries"
print_result_csv "Runtime summary" "${SUMMARY_DIR}/summary.csv"
print_result_csv "Optimization progress" "${SUMMARY_DIR}/optimization_progress.csv"
print_result_csv "Loom vs others summary" "${SUMMARY_DIR}/loom_vs_others_summary.csv"
print_result_csv "Current non-held-out summary" "${SUMMARY_DIR}/current_non_held_out_summary.csv"
print_result_csv "Current CUDA gap" "${SUMMARY_DIR}/current_cuda_vs_best_external_gap.csv"
print_result_csv "Current CUDA top losses" "${SUMMARY_DIR}/current_cuda_top_losses.csv"
print_result_csv "Current secured-win guard" "${SUMMARY_DIR}/current_cuda_secured_win_guard.csv"
print_result_csv "Current Triton gap" "${SUMMARY_DIR}/current_triton_vs_best_fixed_triton_gap.csv"
print_result_csv "Current Triton top losses" "${SUMMARY_DIR}/current_triton_top_losses.csv"
print_result_csv "Current Triton secured-win guard" "${SUMMARY_DIR}/current_triton_secured_win_guard.csv"
FRONTEND_SUMMARY_DIR="${RESULTS_DIR}/frontend/summaries"
print_result_csv "Frontend summary" "${FRONTEND_SUMMARY_DIR}/frontend_summary.csv"
print_result_csv "Frontend parity" "${FRONTEND_SUMMARY_DIR}/frontend_parity.csv"
print_result_csv "Frontend runtime summary" "${FRONTEND_SUMMARY_DIR}/frontend_runtime_summary.csv"
