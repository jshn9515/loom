#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 ]]; then
  set -- --smoke-only
fi

opam exec -- dune build src/loom_cli/main.exe

exec uv run --no-sync python experiments/source/harness/run.py "$@"
