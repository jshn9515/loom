open Loom

let[@loom.entry] l2_norm_sq (x : Tensor1.t) =
  let squares = Tensor1.map (fun xi -> xi *. xi) x in
  Tensor1.reduce_sum squares

