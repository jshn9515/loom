open Loom

let[@loom.entry] dot (x : Tensor1.t) (y : Tensor1.t) =
  let products = Tensor1.map2 (fun xi yi -> xi *. yi) x y in
  Tensor1.reduce_sum products

