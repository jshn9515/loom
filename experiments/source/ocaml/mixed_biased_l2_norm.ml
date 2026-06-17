open Loom

let[@loom.entry] mixed_biased_l2_norm (scale : float) (bias : float)
    (x : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map
       (fun xi ->
         let value = (scale *. xi) +. bias in
         value *. value)
       x)
