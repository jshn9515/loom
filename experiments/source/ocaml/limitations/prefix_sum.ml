open Loom

let[@loom.entry] bad (x : Tensor1.t) =
  let rec prefix value = prefix value in
  Tensor1.map prefix x
