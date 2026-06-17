open Loom

let bad_scalar x = x +. 1.

let[@loom.entry] bad (x : Tensor1.t) =
  Tensor1.map (fun xi -> bad_scalar xi) x
