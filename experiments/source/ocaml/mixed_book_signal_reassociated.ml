open Loom

let[@loom.entry] mixed_book_signal_reassociated (scale : float) (epsilon : float)
    (threshold : float) (bid : Tensor1.t) (ask : Tensor1.t) =
  let signal bid_i ask_i =
    let depth = ask_i +. bid_i in
    let spread = bid_i -. ask_i in
    let normalized = spread /. (epsilon +. depth) in
    if threshold < depth then normalized *. scale else 0.0
  in
  Tensor1.map2 signal bid ask
