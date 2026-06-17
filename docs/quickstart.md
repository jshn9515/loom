# Loom Quickstart

This is the shortest path from OCaml or Python source to either generated
Triton code or a packaged CUDA library.

## Build

Build the compiler:

```sh
make ocaml-build
```

Create the pinned Python runtime environment:

```sh
make python-env
```

## List Entrypoints

Entrypoints must be annotated with `[@loom.entry]` in OCaml or decorated with
`@loom.entry` in Python.

```sh
dune exec ./src/loom_cli/loomc.exe -- list-entries examples/saxpy.ml
```

## Compile

Compile a single entrypoint:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.ml \
  --entry saxpy \
  --target triton \
  --out build/saxpy \
  --emit all
```

Optional compile flags:

- `--target triton|cuda` to choose generated Triton Python or generated CUDA
  shared-library output (`--backend triton|cuda` is an alias)
- `--emit <kind>` to emit a partial artifact family; it defaults to `all`, which
  writes a runnable backend artifact plus IR dumps, manifest, and report
- `--input-kind python` to compile Python source
- `--input-kind front-ir` to compile serialized `FrontIR` JSON directly
- `--autotune` to enable the default Triton autotune policy
- `--autotune-config <path>` to use an explicit Triton autotune config
- `--cuda-arch sm_XX` and `--cuda-platform generic|current` for CUDA output

Triton artifacts are written under `build/saxpy/`:

- `front_ir.json`
- `lambda.sexp`
- `loom_lambda.json`
- `tensor_ir.json`
- `kernel_plan.json`
- `triton_plan.json`
- `backend_analysis.json`
- `pipeline.json`
- `manifest.json`
- `report.md`
- generated Triton Python source

`kernel_plan.json` records backend-neutral plan semantics and body traits.
`triton_plan.json` and `cuda_plan.json` are the target-owned plans that lower
those shared traits into runnable backend artifacts.

Compile the same entrypoint to the generated CUDA backend:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.ml \
  --entry saxpy \
  --target cuda \
  --out build/saxpy-cuda
```

CUDA compile output includes the shared IR dumps, `cuda_plan.json`,
`backend_analysis.json`, generated CUDA source/header files, and `libsaxpy.so`.

Compile the Python spelling of the same entrypoint by selecting the Python
frontend:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.py \
  --input-kind python \
  --entry saxpy \
  --target triton \
  --out build/saxpy-python
```

Compile the C++ spelling by selecting the C++ frontend. The C++ frontend uses
Clang AST parsing and expects the Loom staged subset:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.cpp \
  --input-kind cpp \
  --entry saxpy \
  --target triton \
  --out build/saxpy-cpp
```

## Package

Package a Dune project into a linkable CUDA library:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  package \
  --project examples/package_project \
  --out build/package_project \
  --kind shared
```

Python and C++ package roots can be packaged with `--input-kind python` or
`--input-kind cpp`. A `loom-package.json` file may provide the package name;
otherwise the directory name is used.

Artifacts are written under `build/package_project/`:

- `libpackage_project.so` or `libpackage_project.a`
- `include/loom/package_project.h`
- `src-gen/package_project.cu`
- `manifest.json`
- `report.md`
- per-entry IR dumps under `entries/`, including `kernel_plan.json`,
  `cuda_plan.json`, and `backend_analysis.json`

## Run Tests

```sh
make test
```

Runtime CUDA checks auto-skip when CUDA or Triton is unavailable.

## Run Experiments

Generate the committed experiment outputs:

```sh
make experiments
```

The experiment harness auto-discovers workload configs from
`experiments/source/data/workloads/`, tunes only on the tuning split, evaluates
on held-out sizes, and emits PNG plots whose titles state whether lower or
higher values are better.

Run a single kernel experiment:

```sh
make experiments-kernel KERNEL=saxpy
```
