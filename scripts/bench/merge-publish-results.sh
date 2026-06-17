#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/bench/merge-publish-results.sh --tuning DIR --held-out DIR --frontend DIR --out DIR [--replace]

Combines accepted split DGX artifacts into one public results tree. DIR may be
an unpacked artifact root containing results/, or the results directory itself.
USAGE
}

TUNING_DIR=""
HELD_OUT_DIR=""
FRONTEND_DIR=""
OUT_DIR=""
REPLACE=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tuning)
      TUNING_DIR="$2"
      shift 2
      ;;
    --held-out)
      HELD_OUT_DIR="$2"
      shift 2
      ;;
    --frontend)
      FRONTEND_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --replace)
      REPLACE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TUNING_DIR}" || -z "${HELD_OUT_DIR}" || -z "${FRONTEND_DIR}" || -z "${OUT_DIR}" ]]; then
  usage >&2
  exit 2
fi

resolve_results_dir() {
  local dir="$1"
  if [[ -d "${dir}/results" ]]; then
    printf '%s\n' "${dir}/results"
  else
    printf '%s\n' "${dir}"
  fi
}

TUNING_RESULTS="$(resolve_results_dir "${TUNING_DIR}")"
HELD_OUT_RESULTS="$(resolve_results_dir "${HELD_OUT_DIR}")"
FRONTEND_RESULTS="$(resolve_results_dir "${FRONTEND_DIR}")"

for dir in "${TUNING_RESULTS}" "${HELD_OUT_RESULTS}" "${FRONTEND_RESULTS}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "missing results directory: ${dir}" >&2
    exit 1
  fi
done

if [[ -e "${OUT_DIR}" && "${REPLACE}" -ne 1 ]]; then
  echo "output already exists; pass --replace to overwrite: ${OUT_DIR}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/loom-publish-results.XXXXXX")"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

cp -a "${TUNING_RESULTS}/." "${TMP_DIR}/"
mkdir -p "${TMP_DIR}/frontend"
rm -rf "${TMP_DIR}/frontend"
cp -a "${FRONTEND_RESULTS}/frontend" "${TMP_DIR}/frontend"

python3 - "$TMP_DIR" "$TUNING_RESULTS" "$HELD_OUT_RESULTS" <<'PY'
from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import OrderedDict

out = pathlib.Path(sys.argv[1])
tuning = pathlib.Path(sys.argv[2])
held = pathlib.Path(sys.argv[3])


def merge_csv(relative: str) -> None:
    target = out / relative
    sources = [tuning / relative, held / relative]
    rows = []
    fieldnames = None
    for source in sources:
        if not source.exists():
            continue
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(dict(row) for row in reader)
    if fieldnames is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: pathlib.Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


for relative in [
    "compile/raw/compile_measurements.csv",
    "runtime/raw/runtime_measurements.csv",
    "runtime/raw/tuning_measurements.csv",
    "runtime/raw/verification_measurements.csv",
    "shared/raw/capability_checks.csv",
    "shared/raw/completed_units.csv",
]:
    merge_csv(relative)

manifest_target = out / "shared/raw/dataset_manifest.jsonl"
manifest_target.parent.mkdir(parents=True, exist_ok=True)
with manifest_target.open("w", encoding="utf-8") as handle:
    seen = set()
    for root in [tuning, held]:
        source = root / "shared/raw/dataset_manifest.jsonl"
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line or line in seen:
                continue
            seen.add(line)
            handle.write(line + "\n")

failures = []
for root in [tuning, held]:
    failures.extend(load_json(root / "runtime/raw/verification_failures.json", {"failures": []}).get("failures", []))
(out / "runtime/raw/verification_failures.json").write_text(
    json.dumps({"failures": failures}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

tuning_decisions = OrderedDict()
for root in [tuning, held]:
    payload = load_json(root / "runtime/raw/tuning_decisions.json", {})
    for kernel, decisions in payload.items():
        tuning_decisions.setdefault(kernel, {}).update(decisions)
(out / "runtime/raw/tuning_decisions.json").write_text(
    json.dumps(tuning_decisions, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

run_state = load_json(tuning / "shared/raw/run_state.json", {})
held_state = load_json(held / "shared/raw/run_state.json", {})
kernels = list(dict.fromkeys([*run_state.get("kernels", []), *held_state.get("kernels", [])]))
run_state["kernels"] = kernels
run_state["mode"] = "split-publish"
run_state["benchmark_set"] = "all"
run_state["held_out_artifact_mode"] = held_state.get("mode", "")
(out / "shared/raw/run_state.json").write_text(
    json.dumps(run_state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

uv run --no-sync python experiments/source/harness/run.py \
  --finalize-only \
  --results-dir "${TMP_DIR}"

rm -rf "${OUT_DIR}"
mkdir -p "$(dirname "${OUT_DIR}")"
cp -a "${TMP_DIR}" "${OUT_DIR}"
echo "Wrote merged public results to ${OUT_DIR}"
