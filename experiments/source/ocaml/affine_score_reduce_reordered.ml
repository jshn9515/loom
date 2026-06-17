open Loom

let[@loom.entry] affine_score_reduce_reordered (scale : float) (bias : float)
    (threshold : float) (x : Tensor1.t) =
  let score xi =
    let shifted = bias +. (xi *. scale) in
    if threshold < shifted then shifted else 0.0
  in
  Tensor1.reduce_sum (Tensor1.map score x)
