# loomc

`loomc` is the Stage 1 compiler CLI.

## Commands

### `list-entries`

Print the names of all Loom entry bindings in a source file:

```sh
dune exec ./src/loom_cli/loomc.exe -- list-entries examples/saxpy.ml
```

For Python sources, pass `--input-kind python`:

```sh
dune exec ./src/loom_cli/loomc.exe -- list-entries examples/saxpy.py --input-kind python
```

For C++ sources, pass `--input-kind cpp`:

```sh
dune exec ./src/loom_cli/loomc.exe -- list-entries examples/saxpy.cpp --input-kind cpp
```

### `front-ir`

Emit only frontend lowering output as `FrontIR` JSON:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  front-ir examples/saxpy.py \
  --input-kind python \
  --entry saxpy
```

### `compile`

Compile a single entrypoint:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.ml \
  --entry saxpy \
  --target triton \
  --out build/saxpy \
  --emit all
```

Or compile the same entrypoint through the generated CUDA backend:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  compile examples/saxpy.ml \
  --entry saxpy \
  --target cuda \
  --out build/saxpy-cuda \
  --emit all
```

Optional compile flags:

- `--target triton|cuda`: choose the backend (`--backend triton|cuda` is an
  alias)
- `--emit <kind>`: choose one output family; defaults to `all`, which emits a
  runnable backend artifact plus IRs, manifest, and report
- `--input-kind ocaml|python|cpp|front-ir`: choose whether the input is OCaml
  source, Python source, C++ source, or a serialized `FrontIR` JSON file
- `--autotune`: enable the default Triton autotune policy
- `--autotune-config <path>`: enable Triton autotuning with an explicit config
  file
- `--cuda-arch sm_XX`: override auto-detected GPU architecture for CUDA output
- `--cuda-platform generic|current`: choose generic CUDA codegen constants or
  current-platform CUDA tuning
- `--opt-config <path>`: load pass-local optimizer thresholds and heuristics
- `--enable-opt <id>`: enable one optimization pass
- `--disable-opt <id>`: disable one optimization pass after earlier enables

The generated CUDA backend accepts the same optimization flags and input-kind
surface, but it rejects Triton autotuning flags. Target-specific emit requests
are validated, so `--emit cuda` is rejected for `--target triton` and
`--emit triton` is rejected for `--target cuda`.

C++ input uses Clang AST parsing. `LOOM_CLANGXX` can override the `clang++`
binary used by the C++ frontend.

List the supported optimization IDs:

```sh
dune exec ./src/loom_cli/loomc.exe -- list-opts
```

### `package`

Package a Dune project into a linkable CUDA library:

```sh
dune exec ./src/loom_cli/loomc.exe -- \
  package \
  --project examples/package_project \
  --out build/package_project \
  --kind shared
```

Optional selectors:

- `--input-kind ocaml|python|cpp|auto`: choose whether package scanning reads
  OCaml files, Python files, C++ files, or all active source frontends
- `--module <name>`: package only matching source modules
- `--entry <name>`: package only matching entrypoints
- `--cuda-arch sm_XX`: override auto-detected GPU architecture
- `--cuda-platform generic|current`: choose generic CUDA codegen constants or
  current-platform CUDA tuning
- `--opt-config <path>`: load pass-local optimizer thresholds and heuristics
- `--enable-opt <id>` / `--disable-opt <id>`: apply the same optimization
  surface to packaged builds

## Output

The output directory includes:

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
- `<entry>_triton.py`

`compile --target triton` emits a runnable Python module. Import the generated
`<entry>_triton.py` file in a Python process with PyTorch and Triton available,
then call the generated function with CUDA tensors.

For `compile --target cuda`, the output directory includes:

- `front_ir.json`
- `lambda.sexp`
- `loom_lambda.json`
- `tensor_ir.json`
- `kernel_plan.json`
- `cuda_plan.json`
- `backend_analysis.json`
- `pipeline.json`
- `manifest.json`
- `report.md`
- `<entry>_cuda.cu`
- `<entry>_cuda.h`
- `lib<entry>.so`

`compile --target cuda` emits a runnable shared library with a C ABI plus a
generated header. The manifest records `target_backend: "cuda"`, exported
symbols, workspace symbols, artifact paths, and the resolved CUDA tuning
constants.

`kernel_plan.json` records shared backend-neutral plan semantics and body traits.
The canonical target plans remain `triton_plan.json` and `cuda_plan.json`, which
apply Triton-specific or CUDA-specific lowering choices on top of those shared
traits.

When autotuning is enabled, the manifest and generated Triton module also
record the selected tuning metadata and size buckets.

When optimization flags are enabled, the manifest, pipeline bundle, and report
record the resolved optimization set.

For `package`, the output directory includes:

- `lib<project>.so` or `lib<project>.a`
- `include/loom/<project>.h`
- `src-gen/<project>.cu`
- `manifest.json`
- `report.md`
- `entries/<symbol>/...` shared IR dumps plus `kernel_plan.json`,
  `cuda_plan.json`, and `backend_analysis.json` for each exported entrypoint
