open Loom

let[@loom.entry] mse_sum (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi ->
         let d = xi -. yi in
         d *. d)
       x y)
