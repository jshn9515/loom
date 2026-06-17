# Loom Experiments

This directory contains reproducible benchmark inputs, generated result data,
and the harness used to compare generated Loom kernels against handwritten
Triton and CUDA implementations.

The public experiment suite is split into three CI jobs:

- tuning benchmarks on the current non-held-out set, comparing Loom against
  handwritten Triton and C++/CUDA baselines
- held-out benchmarks on generalization workloads, also comparing against the
  same external baselines
- frontend benchmarks that compare OCaml, Python, and C++ source lowering,
  compile parity, and Loom-only runtime parity

Performance jobs that include external groups explicitly use the canonical
OCaml sources. Python and C++ are evaluated in the separate frontend job.

## Layout

- `source/ocaml/`: canonical OCaml Loom input programs
- `source/python/`: matching naive Python Loom input programs
- `source/cpp/`: matching staged C++ Loom input programs
- `source/triton/naive/`: straightforward Triton baselines
- `source/triton/naive_autotuned/`: straightforward Triton baselines with
  explicit autotune decorators
- `source/triton/optimized_fixed/`: tuned Triton baselines with fixed launch
  choices
- `source/triton/optimized/`: tuned Triton baselines with explicit autotune
- `source/cuda/naive/`: straightforward C++ CUDA baselines
- `source/cuda/optimized/`: tuned C++ CUDA baselines
- `source/data/tuning/`: tuning-only sizes, seeds, and optimizer config path
- `source/data/evaluation/`: held-out evaluation sizes and seeds
- `source/data/workloads/`: discovered workload config files
- `source/data/profiles/`: discovered Loom optimization-profile config files
- `source/harness/`: measurement, verification, plotting, and report code
- `results/local/`: local generated outputs
- `results/public/`: committed reference outputs promoted from DGX CI artifacts
- `_work/`: ignored intermediate artifacts

## Common Runs

Run the current non-held-out performance set:

```sh
uv run python experiments/source/harness/run.py --all --benchmark-set tuning
```

Run the held-out generalization performance set:

```sh
uv run python experiments/source/harness/run.py --all --benchmark-set held-out
```

`--all` runs a fixed-path compile and verification smoke gate for the selected
benchmark set before starting data generation.

Run one kernel's performance suite plus capability checks:

```sh
uv run python experiments/source/harness/run.py --kernel saxpy
```

Run only fixed-path implementations, excluding autotuned groups:

```sh
uv run python experiments/source/harness/run.py \
  --fixed-only \
  --kernel saxpy \
  --size 131072
```

Run exact implementation/kernel/size cases for focused CUDA-vs-external tuning:

```sh
uv run python experiments/source/harness/run.py \
  --benchmark-case loom_cuda_fixed:saxpy:131072 \
  --benchmark-case cuda_optimized:saxpy:131072
```

`--implementation` filters whole implementation groups. `--benchmark-case`
narrows further to `IMPLEMENTATION:KERNEL` or `IMPLEMENTATION:KERNEL:SIZE`.
Compile runs for selected implementation/kernel pairs, while verification and
runtime run only the selected sizes.

## Current CUDA Presets

The harness exposes only current named presets. Historical preset definitions
are intentionally removed from active code; use git history for old reruns.

```sh
uv run python experiments/source/harness/run.py --preset current-cuda-focused
```

`current-cuda-focused` runs a focused CUDA tuning subset, fixed path only, with
short runtime repetitions.

The current CUDA profile uses trait-selected pointwise routing: simple
activations, affine vector updates, affine clamp, and ratio-book bodies choose
scalar or vector lanes from CUDA body traits and published-size thresholds.
Generated CUDA artifacts also expose a no-workspace entrypoint when the CudaPlan
does not require temporary storage; the harness uses that entrypoint to avoid
workspace ABI overhead for pure pointwise kernels while retaining the general
workspace-capable symbol.

```sh
uv run python experiments/source/harness/run.py --preset current-cuda-pass
```

`current-cuda-pass` runs the full current non-held-out CUDA tuning set, fixed
path only, with short runtime repetitions.

```sh
uv run python experiments/source/harness/run.py --preset current-cuda-milestone
```

`current-cuda-milestone` runs the same non-held-out CUDA set with higher
runtime repetitions for milestone checks.

Current CUDA summary outputs are:

- `runtime/summaries/current_non_held_out_summary.csv`
- `runtime/summaries/current_cuda_vs_best_external_gap.csv`
- `runtime/summaries/current_cuda_top_losses.csv`
- `runtime/summaries/current_cuda_secured_win_guard.csv`

Held-out generalization workloads are excluded from current CUDA tuning
presets. They are run only by the held-out publishing job and guard against
backend logic that is accidentally coupled to benchmark entry names instead of
TensorIR/KernelPlan/CudaPlan traits.

## Current Triton Presets

Current Triton presets mirror the CUDA tuning flow but compare generated Loom
Triton against fixed handwritten Triton baselines. They are intended for
Triton-backend-only optimization rounds; do not use them to justify shared
pipeline changes.

```sh
uv run python experiments/source/harness/run.py --preset current-triton-focused
```

`current-triton-focused` runs the current worst Triton fixed-path gap set,
fixed path only, with short runtime repetitions.

```sh
uv run python experiments/source/harness/run.py --preset current-triton-pass
```

`current-triton-pass` runs the full current non-held-out Triton tuning set,
fixed path only, with short runtime repetitions.

```sh
uv run python experiments/source/harness/run.py --preset current-triton-milestone
```

`current-triton-milestone` runs the same non-held-out Triton set with higher
runtime repetitions for milestone checks.

Current Triton summary outputs are:

- `runtime/summaries/current_triton_vs_best_fixed_triton_gap.csv`
- `runtime/summaries/current_triton_top_losses.csv`
- `runtime/summaries/current_triton_secured_win_guard.csv`

## Optimization-Pass Runs

For fast candidate comparison, run only the modified group and an accepted
baseline group:

```sh
uv run python experiments/source/harness/run.py \
  --optimization-pass \
  --candidate-group loom_cuda_fixed \
  --baseline-group loom_cuda_previous_fixed \
  --kernel saxpy
```

Optimization-pass mode forces fixed-path implementations, skips autotuned
groups, defaults to `runtime_repetitions=5` and `compile_repetitions=1`, and
emits `runtime/summaries/optimization_progress.csv`.

Use optimization-pass outputs for self-improvement loops. Use the split tuning
and held-out publishing jobs for public-facing comparisons so all external
performance data is produced on the same hardware setup while keeping frontend
comparisons isolated.

## Frontend Comparison

Run OCaml/Python/C++ frontend lowering without changing backend/public
performance result data:

```sh
uv run python experiments/source/harness/run.py \
  --frontend-comparison \
  --kernel saxpy
```

This writes `frontend/raw/frontend_measurements.csv`,
`frontend/summaries/frontend_summary.csv`, and
`frontend/summaries/frontend_parity.csv`.

For a public credibility run, include isolated runtime parity across the full
frontend suite:

```sh
uv run python experiments/source/harness/run.py \
  --frontend-comparison \
  --frontend-runtime
```

The Python and C++ sources are intentionally naive high-level Loom programs.
The comparison measures source-to-`FrontIR` coverage and Loom backend parity
rather than backend-specific handwritten code.

## Results And Finalization

Run only the pre-publish smoke gate:

```sh
uv run python experiments/source/harness/run.py --smoke-only
```

Regenerate plots from existing CSV outputs:

```sh
uv run python experiments/source/harness/run.py --plots-only
```

Rebuild summaries, plots, the dataset manifest, and the report from persisted
raw outputs:

```sh
uv run python experiments/source/harness/run.py --finalize-only
```

Outputs are written to `results/local/` by default, split into `compile/`,
`runtime/`, `frontend/`, and `shared/`. Compiled intermediates go to `_work/`.
Reproducibility comes from fixed seeds and config files under `source/data/`.

Raw measurement writers buffer rows during an experiment unit and flush durably
when that unit completes. This keeps long runs recoverable without `fsync`-ing
every appended row. The completion ledger is written to:

- `shared/raw/completed_units.csv`
- `shared/raw/run_state.json`

If a run is interrupted after unit flushes, `--finalize-only` can rebuild final
summaries, plots, manifests, and reports from the persisted raw files.

Accepted split artifacts are merged into one public result tree with:

```sh
scripts/bench/merge-publish-results.sh \
  --tuning <unpacked-tuning-artifact> \
  --held-out <unpacked-held-out-artifact> \
  --frontend <unpacked-frontend-artifact> \
  --out experiments/results/public \
  --replace
```

## DGX Container And Gitea CI

DGX benchmark runs execute on a Gitea self-hosted runner with Docker and NVIDIA
Container Runtime. Development changes are pushed to `dev`; the DGX runner
builds the benchmark image locally before running the harness. The workflow is
disabled on `master`, which is reserved for the squashed public tree and
accepted public plots.

Local helper to push the current commit to `dev`:

```sh
scripts/bench/tea-dev-benchmark.sh
```

Build the benchmark image manually on the DGX:

```sh
scripts/bench/build-image.sh
```

The benchmark image uses the stable local tag `loom-bench:dev` by default. The
build script removes the previous stable tag when no container still references
it, preserving shared layers and BuildKit cache for faster rebuilds.

Run a containerized smoke test:

```sh
scripts/bench/run-container.sh --smoke-only
```

Run a partial benchmark:

```sh
scripts/bench/run-container.sh --kernel saxpy --size 131072 --runtime-repetitions 5
```

Run a current CUDA preset through the container:

```sh
scripts/bench/run-container.sh --preset current-cuda-pass
```

Run a current Triton preset through the container:

```sh
scripts/bench/run-container.sh --preset current-triton-pass
```

CI writes DGX logs, results, and work files under
`experiments/artifacts/dgx/<sha>/<run-id>/`. Uploaded workflow artifacts are
trimmed to logs, result files, and smoke-gate diagnostics; generated work
directories remain runner-local. Promotion into `experiments/results/public/`
is intentionally manual.

Manual workflow dispatch from `dev` is split by workflow so push events cannot
enqueue public publishing jobs. The `DGX Smoke` workflow accepts:

- `mode=smoke`
- `mode=partial` plus `harness_args`, for example
  `--kernel saxpy --size 131072`

The `DGX Publish Benchmarks` workflow accepts:

- `mode=tuning` for the non-held-out performance job only
- `mode=held-out` for the held-out performance job only
- `mode=frontend` for the frontend comparison job only
- `mode=publish` to run tuning, held-out, and frontend jobs as separate CI jobs

When local public results are accepted, publish `dev` to `master` as one
squashed public commit:

```sh
scripts/bench/publish-master-from-dev.sh -m "Publish public benchmark results"
```

## Protocol Details

The current publication protocol is:

1. push to `dev`; push-triggered smoke verifies the runner and fixed-path gate
2. dispatch `mode=publish` on `dev`
3. tuning job runs `--all --benchmark-set tuning`
4. held-out job runs `--all --benchmark-set held-out`
5. frontend job runs `--frontend-comparison --frontend-runtime`
6. download and inspect all three artifacts
7. merge accepted artifacts with `scripts/bench/merge-publish-results.sh`
8. review generated public plots and summaries before publishing to `master`

Dataset splits are deterministic and config-driven:

- `source/data/tuning/settings.json` defines tuning sizes and seed offsets
- `source/data/evaluation/settings.json` defines evaluation sizes and seed offsets
- `configs/optimizations/current.json` defines the active shared optimizer config

The audit step checks for configuration drift between workload metadata, the
catalog in `source/data/common/workloads.json`, PyTorch reference coverage, and
the presence of Triton/CUDA baseline source files. It also writes a
generalizability audit for CUDA backend name-sensitive dispatch. Outputs are:

- `shared/raw/audit_report.json`
- `shared/raw/generalizability_audit.json`

The harness emits two plot families for both compile and runtime:

- `compile/plots/loom_internal/` and `runtime/plots/loom_internal/`
- `compile/plots/loom_vs_others/` and `runtime/plots/loom_vs_others/`

Each run also writes `shared/raw/environment.json` and
`shared/raw/environment.txt` with GPU model, driver version, CUDA runtime
information, and raw `nvidia-smi` output.
