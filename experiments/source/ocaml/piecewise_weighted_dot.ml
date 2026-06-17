open Loom

let[@loom.entry] piecewise_weighted_dot (weight_pos : float) (weight_neg : float) (x : Tensor1.t)
    (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi ->
         let prod = xi *. yi in
         if xi > 0.0 then weight_pos *. prod else weight_neg *. prod)
       x y)
