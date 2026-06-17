open Loom

let[@loom.entry] mixed_weighted_affine_dot (weight : float) (scale : float)
    (bias : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi ->
         let transformed = (scale *. xi) +. bias in
         weight *. transformed *. yi)
       x y)
