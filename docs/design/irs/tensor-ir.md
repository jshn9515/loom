# TensorIR

`TensorIR` is Loom's first optimization-facing tensor graph IR.

It is the target of `LoomLambda -> TensorIR`.

## Purpose

- lower semantic tensor combinators into explicit kernelable nodes
- expose elementwise and reduction operations directly
- define the first optimization insertion stage

## Shape

`TensorIR` currently models:

- rank-1 `float32` tensor params
- scalar float params
- elementwise 1D nodes
- reduction 1D nodes
- scalar expressions for elementwise bodies

## Serialization

The compiler emits `tensor_ir.json`.

Reference schema: `docs/design/irs/schemas/tensor-ir.schema.json`

Top-level JSON shape:

```json
{
  "entry_name": "saxpy",
  "params": [...],
  "result": { "kind": "tensor", "value": "node_0" },
  "nodes": [...]
}
```

Node kinds:

- `"elementwise1d"`
- `"reduce1d"`

## Invariants

- no general lambdas or applications remain
- only kernelable tensor structure remains
- scalar bodies are represented with `scalar_expr`
- Stage 1 still has no fusion across nodes
