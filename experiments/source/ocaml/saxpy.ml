open Loom

let[@loom.entry] saxpy (a : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.map2 (fun xi yi -> a *. xi +. yi) x y

