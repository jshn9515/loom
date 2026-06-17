open Loom

let[@loom.entry] bad (x : Tensor1.t) =
  let rec loop y = Tensor1.map (fun xi -> xi) y in
  loop x

