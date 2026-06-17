open Loom

let[@loom.entry] piecewise_weighted_dot_reordered (weight_pos : float)
    (weight_neg : float) (x : Tensor1.t) (y : Tensor1.t) =
  let contribution xi yi =
    let prod = yi *. xi in
    let weight = if 0.0 < xi then weight_pos else weight_neg in
    prod *. weight
  in
  Tensor1.reduce_sum (Tensor1.map2 contribution x y)
