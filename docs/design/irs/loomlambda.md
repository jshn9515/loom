# LoomLambda

`LoomLambda` is Loom's canonical semantic core.

It is the target of `FrontIR -> LoomLambda` normalization.

## Purpose

- represent staged computations in a lambda-calculus-enabled core
- eliminate frontend-specific structure
- expose explicit lambdas, applications, lets, scalar primitives, and tensor
  primitives

## Shape

`LoomLambda` supports:

- `Var`
- `Let`
- `If`
- `Lambda`
- `Apply`
- scalar primitives
- tensor primitives

It does not carry tuple structure.

## Serialization

The compiler emits `loom_lambda.json`.

Reference schema: `docs/design/irs/schemas/loomlambda.schema.json`

Top-level JSON shape:

```json
{
  "entry": "name",
  "params": [{ "name": "a", "type": "float" }],
  "return_type": "tensor1<f32>",
  "body": { "kind": "tensor-prim", "op": "tensor-map2", "args": [...] }
}
```

Expression encodings use:

- `"var"`
- `"float"`
- `"bool"`
- `"let"`
- `"if"`
- `"lambda"`
- `"apply"`
- `"prim"`
- `"tensor-prim"`

## Invariants

- no tuple expressions remain
- frontend helper structure is normalized into explicit lambdas/apps/lets
- local helper calls that can be normalized away should already be reduced here
- tensor combinators remain explicit for the next stage
