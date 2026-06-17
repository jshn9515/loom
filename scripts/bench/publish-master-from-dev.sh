#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bench/publish-master-from-dev.sh [--dry-run] [--remote REMOTE] [--dev BRANCH] [--master BRANCH] [-m MESSAGE]

Publishes the current dev branch tree as one squashed commit on top of master.
This keeps master readable while preserving detailed optimization history on dev.
USAGE
}

REMOTE="origin"
DEV_BRANCH="dev"
MASTER_BRANCH="master"
MESSAGE="Publish public Loom benchmark state"
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --dev)
      DEV_BRANCH="$2"
      shift 2
      ;;
    --master)
      MASTER_BRANCH="$2"
      shift 2
      ;;
    -m|--message)
      MESSAGE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree must be clean before publishing" >&2
  exit 1
fi

git fetch "${REMOTE}" "${DEV_BRANCH}" "${MASTER_BRANCH}"

DEV_REF="${REMOTE}/${DEV_BRANCH}"
MASTER_REF="${REMOTE}/${MASTER_BRANCH}"
DEV_SHA="$(git rev-parse "${DEV_REF}")"
MASTER_SHA="$(git rev-parse "${MASTER_REF}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/loom-publish-master.XXXXXX")"
cleanup() {
  git worktree remove --force "${TMP_DIR}" >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

git worktree add --detach "${TMP_DIR}" "${MASTER_SHA}" >/dev/null
git -C "${TMP_DIR}" read-tree --reset -u "${DEV_SHA}"

if [[ -z "$(git -C "${TMP_DIR}" status --porcelain)" ]]; then
  echo "master already matches dev tree at ${DEV_SHA}"
  exit 0
fi

git -C "${TMP_DIR}" commit -m "${MESSAGE}" >/dev/null
PUBLISH_SHA="$(git -C "${TMP_DIR}" rev-parse HEAD)"

echo "remote=${REMOTE}"
echo "dev=${DEV_REF} (${DEV_SHA})"
echo "master_base=${MASTER_REF} (${MASTER_SHA})"
echo "publish_commit=${PUBLISH_SHA}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: would run git push ${REMOTE} ${PUBLISH_SHA}:refs/heads/${MASTER_BRANCH}"
  exit 0
fi

git push "${REMOTE}" "${PUBLISH_SHA}:refs/heads/${MASTER_BRANCH}"
echo "Published ${PUBLISH_SHA} to ${REMOTE}/${MASTER_BRANCH}."
