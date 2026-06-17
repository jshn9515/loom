open Loom

let[@loom.entry] affine_clamp (scale : float) (bias : float) (lo : float) (hi : float)
    (x : Tensor1.t) =
  Tensor1.map
    (fun xi ->
      let value = (scale *. xi) +. bias in
      if value < lo then lo else if value > hi then hi else value)
    x
