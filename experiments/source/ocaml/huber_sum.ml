open Loom

let[@loom.entry] huber_sum (delta : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun xi yi ->
         let d = xi -. yi in
         if d > delta then
           delta *. (d -. (0.5 *. delta))
         else if d < (-.delta) then
           delta *. ((-.d) -. (0.5 *. delta))
         else
           0.5 *. d *. d)
       x y)
