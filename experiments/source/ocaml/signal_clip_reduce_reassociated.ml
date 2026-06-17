open Loom

let[@loom.entry] signal_clip_reduce_reassociated (scale : float) (epsilon : float)
    (clip : float) (bid : Tensor1.t) (ask : Tensor1.t) =
  let contribution bid_i ask_i =
    let depth = ask_i +. bid_i in
    let raw = ((bid_i -. ask_i) /. (depth +. epsilon)) *. scale in
    let neg_clip = 0.0 -. clip in
    if raw < neg_clip then neg_clip else if clip < raw then clip else raw
  in
  Tensor1.reduce_sum (Tensor1.map2 contribution bid ask)
