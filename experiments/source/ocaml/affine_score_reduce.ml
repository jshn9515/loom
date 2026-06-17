open Loom

let[@loom.entry] affine_score_reduce (scale : float) (bias : float) (threshold : float)
    (x : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map
       (fun xi ->
         let value = (scale *. xi) +. bias in
         if value > threshold then value else 0.0)
       x)
