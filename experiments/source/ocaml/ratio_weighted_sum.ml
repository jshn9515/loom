open Loom

let[@loom.entry] ratio_weighted_sum (scale : float) (epsilon : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi -> (scale *. xi) /. (yi +. epsilon))
       x y)
