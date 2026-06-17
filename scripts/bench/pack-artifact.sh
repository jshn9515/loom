#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:?artifact root required}"
ARTIFACT_PACK="${2:?artifact pack path required}"

mkdir -p "$(dirname "${ARTIFACT_PACK}")"
if [[ -d "${ARTIFACT_ROOT}" ]]; then
  pack_items=()
  for item in logs results; do
    if [[ -e "${ARTIFACT_ROOT}/${item}" ]]; then
      pack_items+=("${item}")
    fi
  done
  if [[ -d "${ARTIFACT_ROOT}/work/smoke_gate_results" ]]; then
    pack_items+=("work/smoke_gate_results")
  fi
  if [[ "${#pack_items[@]}" -gt 0 ]]; then
    tar -czf "${ARTIFACT_PACK}" -C "${ARTIFACT_ROOT}" "${pack_items[@]}"
  else
    printf 'Benchmark artifact root had no packable outputs: %s\n' "${ARTIFACT_ROOT}" > "${ARTIFACT_PACK}.missing.txt"
  fi
else
  printf 'No benchmark artifact root found: %s\n' "${ARTIFACT_ROOT}" > "${ARTIFACT_PACK}.missing.txt"
fi
