# KernelPlan

`KernelPlan` is the shared backend-oriented launch and trait plan that sits
between `TensorIR` and backend-owned plans.

The active backend contracts are:

- `TensorIR -> KernelPlan traits -> TritonPlan -> Triton`
- `TensorIR -> KernelPlan traits -> CudaPlan -> CUDA`

`kernel_plan.json` is emitted for compatibility and for inspecting shared
semantics before target-specific planning.

## Purpose

- choose kernel boundaries
- assign temporary names
- attach launch metadata
- classify backend-neutral scalar body traits for target-specific backends
- preserve a deterministic shared planning view during backend specialization

## Shape

`KernelPlan` currently models:

- elementwise steps
- reduction steps
- result naming
- temporary counts
- fixed launch parameters from the shared planner
- body traits such as dot, weighted, square, ratio, branch, clip, and
  pipeline-expanded

## Serialization

The compiler emits `kernel_plan.json` as a shared artifact.

Reference schema: `docs/design/irs/schemas/kernel-plan.schema.json`

Top-level JSON shape:

```json
{
  "entry_name": "dot",
  "steps": [...],
  "result_name": "out",
  "temporary_count": 1
}
```

Step kinds:

- `"elementwise"`
- `"reduction"`

## Current Status

- `KernelPlan` holds backend-neutral semantics and traits, not target scheduling
  policy
- target launch and codegen choices belong in `TritonPlan` and `CudaPlan`
- both generated Triton and generated CUDA may consume shared traits when
  selecting backend-specific plans
