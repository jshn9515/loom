#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE_NAME="${LOOM_BENCH_IMAGE_NAME:-loom-bench}"
IMAGE_TAG="${LOOM_BENCH_IMAGE_TAG:-dev}"
DOCKERFILE="${LOOM_BENCH_DOCKERFILE:-containers/bench/Dockerfile}"
GIT_SHA="$(git rev-parse --short=12 HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if docker ps -a --filter "ancestor=${IMAGE}" --format '{{.ID}}' | grep -q .; then
    echo "cannot replace ${IMAGE}: one or more containers still reference it" >&2
    echo "remove those containers or set LOOM_BENCH_IMAGE_TAG to a different tag" >&2
    exit 1
  fi
  docker rmi "${IMAGE}" >/dev/null
fi

docker build \
  --label "org.opencontainers.image.source=$(git config --get remote.origin.url || true)" \
  --label "org.opencontainers.image.revision=${GIT_SHA}" \
  --label "com.loom.branch=${GIT_BRANCH}" \
  --tag "${IMAGE}" \
  --file "${DOCKERFILE}" \
  .

echo "${IMAGE}"
