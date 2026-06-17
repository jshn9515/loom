open Loom

let[@loom.entry] quote_filter (threshold : float) (bid : Tensor1.t) (ask : Tensor1.t) =
  Tensor1.map2
    (fun bi ai ->
      let spread = ai -. bi in
      if spread > threshold then spread else 0.)
    bid ask
