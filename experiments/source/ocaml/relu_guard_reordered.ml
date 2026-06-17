open Loom

let[@loom.entry] relu_guard_reordered (x : Tensor1.t) =
  let activate xi = if 0.0 < xi then xi else 0.0 in
  Tensor1.map activate x
