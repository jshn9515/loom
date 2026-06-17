open Loom

let[@loom.entry] bad (x : Tensor1.t) =
  Tensor1.map (fun xi -> xi +. Tensor1.reduce_sum x) x

