open Loom

let[@loom.entry] relu (x : Tensor1.t) =
  Tensor1.map (fun xi -> if xi > 0. then xi else 0.) x

