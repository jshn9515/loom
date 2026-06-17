open Loom

let[@loom.entry] clipped_huber_sum (delta : float) (cap : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi ->
         let d = xi -. yi in
         let abs_d = if d >= 0.0 then d else -.d in
         let quadratic = 0.5 *. d *. d in
         let linear = delta *. (abs_d -. (0.5 *. delta)) in
         let value = if abs_d > delta then linear else quadratic in
         if value > cap then cap else value)
       x y)
