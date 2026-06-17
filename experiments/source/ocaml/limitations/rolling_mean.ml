open Loom

let[@loom.entry] bad (x : Tensor1.t) =
  let rec rolling value = rolling value in
  Tensor1.map rolling x
