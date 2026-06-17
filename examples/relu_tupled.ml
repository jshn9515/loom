open Loom

let[@loom.entry] relu_tupled (x : Tensor1.t) =
  let thresholds = (0., 1.) in
  let zero, _one = thresholds in
  let clip xi = if xi > zero then xi else zero in
  Tensor1.map clip x
