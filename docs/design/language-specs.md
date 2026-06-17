# Loom Stage 1 Language Spec

## Entrypoints

Loom compiles only top-level OCaml bindings annotated with `[@loom.entry]`,
top-level Python functions decorated with `@loom.entry`, and top-level C++
functions marked with `LOOM_ENTRY`.

```ocaml
open Loom

let[@loom.entry] saxpy (a : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.map2 (fun xi yi -> a *. xi +. yi) x y
```

```python
import loom


@loom.entry
def saxpy(a: float, x: loom.Tensor1, y: loom.Tensor1) -> loom.Tensor1:
    return loom.Tensor1.map2(lambda xi, yi: a * xi + yi, x, y)
```

```cpp
#include <loom/loom.hpp>

LOOM_ENTRY loom::Tensor1 saxpy(float a, loom::Tensor1 x, loom::Tensor1 y) {
  auto blend = [a](float xi, float yi) -> float { return (a * xi) + yi; };
  return loom::Tensor1::map2(blend, x, y);
}
```

## Supported Types

- `float`
- `bool` in scalar control flow
- `Loom.Tensor1.t` / `loom.Tensor1` / `loom::Tensor1`

All staged numeric values lower to backend `float32`.

## Tensor API

Stage 1 exposes the same tensor API in all source frontends:

```ocaml
module Tensor1 : sig
  type t

  val map : (float -> float) -> t -> t
  val map2 : (float -> float -> float) -> t -> t -> t
  val reduce_sum : t -> float
  val reduce_max : t -> float
end
```

Python spells the same API as `loom.Tensor1.map`, `map2`, `reduce_sum`, and
`reduce_max`.

C++ spells the same API as `loom::Tensor1::map`, `map2`, `reduce_sum`, and
`reduce_max`.

## Supported Constructs

- scalar `let`
- tensor `let`
- local non-recursive helper functions
- partial application of local helpers when normalization can eliminate it
- pure float arithmetic
- scalar comparisons
- scalar `if ... then ... else ...`
- tuple construction and tuple destructuring in local staged setup code
- `Tensor1.map`
- `Tensor1.map2`
- `Tensor1.reduce_sum`
- `Tensor1.reduce_max`
- scalar captures inside map lambdas

## Rejected Constructs

- recursion
- mutation and references
- exceptions and IO
- arbitrary external calls inside staged regions
- tuples in staged tensor results
- tensor capture inside scalar lambdas

## Python Frontend Notes

The Python frontend is source-to-`FrontIR`; Python code is parsed but not
executed. Entrypoint parameters and returns must use annotations so the
frontend can produce typed `FrontIR`. Local helper functions should annotate
parameters and returns when they are passed to tensor combinators or used
through partial application. The supported subset intentionally keeps benchmark
source naive and high-level rather than exposing backend-specific constructs.

## C++ Frontend Notes

The C++ frontend is source-to-`FrontIR` through Clang AST JSON; C++ code is
parsed but not executed. It supports Loom's staged C++ subset, not arbitrary
C++: entrypoints use `LOOM_ENTRY`, tensor values use `loom::Tensor1`, tuple
setup uses `loom::tuple` plus structured bindings, and partial application uses
`loom::partial`. The parser is selected with `LOOM_CLANGXX` or defaults to
`clang++`.
