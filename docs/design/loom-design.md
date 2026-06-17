# Loom Design

Loom is currently organized as a staged compiler pipeline:

1. frontend-specific lowering into `FrontIR`
2. normalization from `FrontIR` into `LoomLambda`
3. first optimization insertion stage: `LoomLambda -> TensorIR`
4. shared backend planning from `TensorIR` into KernelPlan semantics and body
   traits
5. backend-local planning and lowering:
   - `KernelPlan traits -> TritonPlan -> Triton`
   - `KernelPlan traits -> CudaPlan -> CUDA`

The OCaml-specific lowering code lives under `src/frontends/ocaml/`, the
Python AST frontend lives under `src/frontends/python/`, and the Clang-AST C++
frontend lives under `src/frontends/cpp/`. All source frontends emit the same
typed `FrontIR`, and the IR and backend logic stays under the Loom-specific
libraries.

## IR Stack

- [`irs/front-ir.md`](irs/front-ir.md)
- [`irs/loomlambda.md`](irs/loomlambda.md)
- [`irs/tensor-ir.md`](irs/tensor-ir.md)
- [`irs/kernel-plan.md`](irs/kernel-plan.md)
- [`optimizations.md`](optimizations.md)

## Transparency

Every successful compile emits inspectable artifacts:

- `FrontIR` JSON
- raw Lambda dump
- `LoomLambda` JSON
- `TensorIR` JSON
- shared KernelPlan JSON plus backend plan JSON (`triton_plan.json` or
  `cuda_plan.json`)
- backend analysis JSON
- pipeline bundle JSON
- generated Triton Python
- manifest
- report

## Stage 1 Shape and Type Model

- only rank-1 tensors
- only `float32`
- only contiguous CUDA tensors at runtime
- `map` and `map2` preserve shape
- `reduce_sum` and `reduce_max` return a scalar represented as a one-element
  `torch.float32` tensor

## Optimization Boundaries

- `FrontIR -> LoomLambda`: normalization-time simplification only
- `LoomLambda -> TensorIR`: first optimization insertion stage
- `TensorIR -> KernelPlan traits`: shared backend-neutral body and launch
  semantics
- `KernelPlan traits -> backend plan`: target-specific Triton or CUDA planning
- backend plan -> codegen: backend-local lowering only

The current first-pass optimization set is documented in
[`optimizations.md`](optimizations.md) and exposed through explicit CLI flags.

## Current Tree

- `src/frontends/ocaml/`: OCaml parsing, entry discovery, and lowering to `FrontIR`
- `src/frontends/python/`: Python AST entry discovery and lowering to `FrontIR`
- `src/frontends/cpp/`: Clang-AST C++ entry discovery and lowering to `FrontIR`
- `src/loom_core/`: core IRs, shared KernelPlan semantics, and shared body traits
- `src/loom_normalize/`: `FrontIR -> LoomLambda`
- `src/loom_tensorize/`: `LoomLambda -> TensorIR`
- `src/loom_backend_triton/`: `TensorIR + KernelPlan traits -> TritonPlan -> Triton`
- `src/loom_backend_cuda/`: `TensorIR + KernelPlan traits -> CudaPlan -> CUDA`
