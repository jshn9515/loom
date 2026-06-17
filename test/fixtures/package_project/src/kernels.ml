open Loom

let[@loom.entry] saxpy (a : float) (x : Tensor1.t) (y : Tensor1.t) =
  Tensor1.map2 (fun xi yi -> a *. xi +. yi) x y

let[@loom.entry] relu (x : Tensor1.t) =
  Tensor1.map (fun xi -> if xi > 0. then xi else 0.) x

let[@loom.entry] dot (x : Tensor1.t) (y : Tensor1.t) =
  let products = Tensor1.map2 (fun xi yi -> xi *. yi) x y in
  Tensor1.reduce_sum products
