open Loom

let[@loom.entry] saxpy_curried (a : float) (x : Tensor1.t) (y : Tensor1.t) =
  let blend a xi yi = a *. xi +. yi in
  Tensor1.map2 (blend a) x y
