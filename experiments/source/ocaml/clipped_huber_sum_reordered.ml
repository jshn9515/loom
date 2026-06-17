open Loom

let[@loom.entry] clipped_huber_sum_reordered (delta : float) (cap : float)
    (x : Tensor1.t) (y : Tensor1.t) =
  let contribution xi yi =
    let d = xi -. yi in
    let abs_d = if d < 0.0 then 0.0 -. d else d in
    let half_delta = 0.5 *. delta in
    let quadratic = (d *. d) *. 0.5 in
    let linear = delta *. (abs_d -. half_delta) in
    let value = if delta < abs_d then linear else quadratic in
    if cap < value then cap else value
  in
  Tensor1.reduce_sum (Tensor1.map2 contribution x y)
