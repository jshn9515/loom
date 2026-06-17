open Loom

let[@loom.entry] soft_threshold (threshold : float) (x : Tensor1.t) =
  Tensor1.map
    (fun xi ->
      if xi > threshold then
        xi -. threshold
      else if xi < (-.threshold) then
        xi +. threshold
      else
        0.)
    x
