open Loom

let[@loom.entry] dot_pipeline (x : Tensor1.t) (y : Tensor1.t) =
  let mul xi yi = xi *. yi in
  let reduce t = Tensor1.reduce_sum t in
  reduce (Tensor1.map2 mul x y)
