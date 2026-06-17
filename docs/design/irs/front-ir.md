# FrontIR

`FrontIR` is Loom's public frontend interchange format.

Source frontends lower into `FrontIR`, not directly into `LoomLambda` or
`TensorIR`. The OCaml, Python, and C++ frontends all use this boundary.

## Purpose

- provide a typed, language-neutral staged program format
- preserve frontend structure that normalization still needs
- carry richer source constructs than `LoomLambda`, especially:
  - tuple construction and tuple destructuring
  - local helper lambdas and applications
  - direct tensor combinator calls

## Shape

`FrontIR` currently supports:

- typed variables
- `let`
- `if`
- lambdas and applications
- scalar primitives
- tensor primitives
- tuples and tuple patterns

## Serialization

The compiler emits `front_ir.json`.

Reference schema: `docs/design/irs/schemas/front-ir.schema.json`

Top-level JSON shape:

```json
{
  "entry": "name",
  "params": [{ "name": "x", "type": "tensor1<f32>" }],
  "return_type": "tensor1<f32>",
  "body": { "kind": "tensor-prim", "op": "tensor-map", "args": [...] }
}
```

Expression encodings use a `"kind"` discriminator such as:

- `"var"`
- `"float"`
- `"bool"`
- `"unit"`
- `"tuple"`
- `"let"`
- `"if"`
- `"lambda"`
- `"apply"`
- `"prim"`
- `"tensor-prim"`

Patterns use:

- `"var"`
- `"tuple"`

## Invariants

- all nodes are typed with Loom stage types
- tensor primitives remain explicit
- tuple structure may still be present
- helper lambdas/applications may still be present
- no source-language syntax should survive beyond what this IR models directly
