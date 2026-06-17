open Loom

let[@loom.entry] book_imbalance (epsilon : float) (bid : Tensor1.t) (ask : Tensor1.t) =
  Tensor1.map2
    (fun bi ai -> (bi -. ai) /. (bi +. ai +. epsilon))
    bid ask
