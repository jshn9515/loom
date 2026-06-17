#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bench/tea-dev-benchmark.sh [--dry-run] [--remote REMOTE] [--branch BRANCH]

Verifies local Gitea/tea access, then pushes HEAD to the benchmark branch.
The DGX Gitea runner handles actual benchmark execution.
USAGE
}

REMOTE="origin"
BRANCH="dev"
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
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v tea >/dev/null 2>&1; then
  echo "tea CLI is required on the dev machine" >&2
  exit 127
fi

if ! tea login list >/dev/null; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "warning: tea is installed but not configured; dry-run continuing" >&2
  else
    echo "tea is not configured; run tea login add before pushing to dev" >&2
    exit 1
  fi
fi

REMOTE_URL="$(git config --get "remote.${REMOTE}.url")"
HEAD_SHA="$(git rev-parse HEAD)"

echo "remote=${REMOTE_URL}"
echo "target=${REMOTE}/${BRANCH}"
echo "head=${HEAD_SHA}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: would run git push ${REMOTE} HEAD:${BRANCH}"
  exit 0
fi

git push "${REMOTE}" "HEAD:${BRANCH}"

echo "Pushed ${HEAD_SHA} to ${REMOTE}/${BRANCH}."
echo "Open the Gitea Actions page for ${REMOTE_URL} to watch the DGX benchmark job."
