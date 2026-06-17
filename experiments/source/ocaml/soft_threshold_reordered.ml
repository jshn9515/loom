open Loom

let[@loom.entry] soft_threshold_reordered (threshold : float) (x : Tensor1.t) =
  let shrink xi =
    if xi < (0.0 -. threshold) then xi +. threshold
    else if threshold < xi then xi -. threshold
    else 0.0
  in
  Tensor1.map shrink x
