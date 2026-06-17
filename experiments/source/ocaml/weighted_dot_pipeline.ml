open Loom

let[@loom.entry] weighted_dot_pipeline (weight : float) (x : Tensor1.t) (y : Tensor1.t) =
  let weighted_term xi yi =
    let prod = xi *. yi in
    weight *. prod
  in
  Tensor1.reduce_sum (Tensor1.map2 weighted_term x y)
