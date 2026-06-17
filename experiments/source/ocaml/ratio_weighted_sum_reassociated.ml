open Loom

let[@loom.entry] ratio_weighted_sum_reassociated (scale : float) (epsilon : float)
    (x : Tensor1.t) (y : Tensor1.t) =
  let contribution xi yi =
    let scaled = xi *. scale in
    scaled /. (epsilon +. yi)
  in
  Tensor1.reduce_sum (Tensor1.map2 contribution x y)
