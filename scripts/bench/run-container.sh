#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bench/run-container.sh [--image IMAGE] [--results-dir DIR] [--work-dir DIR] [--] [HARNESS_ARGS...]

Runs the Loom benchmark harness inside the local benchmark container.
If no harness args are provided, defaults to --smoke-only.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GIT_SHA="$(git rev-parse --short=12 HEAD)"
RUN_ID="${GITEA_RUN_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
IMAGE="${LOOM_BENCH_IMAGE_NAME:-loom-bench}:${LOOM_BENCH_IMAGE_TAG:-dev}"
RESULTS_DIR=""
WORK_DIR=""
HARNESS_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --)
      shift
      HARNESS_ARGS+=("$@")
      break
      ;;
    *)
      HARNESS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#HARNESS_ARGS[@]}" -eq 0 ]]; then
  HARNESS_ARGS=(--smoke-only)
fi

if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="experiments/results/dgx/${GIT_SHA}/${RUN_ID}"
fi
if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR="experiments/_work/dgx/${GIT_SHA}/${RUN_ID}"
fi

has_arg() {
  local needle="$1"
  shift
  local value
  for value in "$@"; do
    if [[ "${value}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

mkdir -p "${RESULTS_DIR}" "${WORK_DIR}" ".container-cache"

HOST_RESULTS_DIR="$(realpath "${RESULTS_DIR}")"
HOST_WORK_DIR="$(realpath "${WORK_DIR}")"

if ! has_arg "--results-dir" "${HARNESS_ARGS[@]}"; then
  HARNESS_ARGS+=(--results-dir "/workspace/${RESULTS_DIR}")
fi
if ! has_arg "--work-dir" "${HARNESS_ARGS[@]}"; then
  HARNESS_ARGS+=(--work-dir "/workspace/${WORK_DIR}")
fi

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
  "${IMAGE}" \
  "${HARNESS_ARGS[@]}"

docker run --rm \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e RESULTS_DIR="/workspace/${RESULTS_DIR}" \
  -e WORK_DIR="/workspace/${WORK_DIR}" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'chown -R "${HOST_UID}:${HOST_GID}" "${RESULTS_DIR}" "${WORK_DIR}" 2>/dev/null || true' \
  >/dev/null 2>&1 || true

METADATA_DIR="${HOST_RESULTS_DIR}/shared/raw"
mkdir -p "${METADATA_DIR}"
HARNESS_ARGS_TEXT="$(printf '%q ' "${HARNESS_ARGS[@]}" | sed 's/[[:space:]]*$//')"
HARNESS_ARGS_JSON="${HARNESS_ARGS_TEXT//\\/\\\\}"
HARNESS_ARGS_JSON="${HARNESS_ARGS_JSON//\"/\\\"}"
{
  echo "{"
  echo "  \"git_sha\": \"$(git rev-parse HEAD)\","
  echo "  \"git_branch\": \"$(git rev-parse --abbrev-ref HEAD)\","
  echo "  \"image\": \"${IMAGE}\","
  echo "  \"image_id\": \"$(docker image inspect "${IMAGE}" --format '{{.Id}}')\","
  echo "  \"run_id\": \"${RUN_ID}\","
  echo "  \"results_dir\": \"${HOST_RESULTS_DIR}\","
  echo "  \"work_dir\": \"${HOST_WORK_DIR}\","
  echo "  \"harness_args\": \"${HARNESS_ARGS_JSON}\","
  echo "  \"docker_version\": $(docker version --format '{{json .}}' 2>/dev/null || echo '{}'),"
  echo "  \"uname\": \"$(uname -a | sed 's/"/\\"/g')\""
  echo "}"
} > "${METADATA_DIR}/container_run.json"

nvidia-smi > "${METADATA_DIR}/container_nvidia_smi.txt" 2>&1 || true
