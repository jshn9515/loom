open Loom

let[@loom.entry] weighted_dot (weight : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi -> weight *. xi *. yi)
       x y)
