open Loom

let[@loom.entry] mixed_book_signal (scale : float) (epsilon : float)
    (threshold : float) (bid : Tensor1.t) (ask : Tensor1.t) =
  Tensor1.map2
    (fun bid_i ask_i ->
      let depth = bid_i +. ask_i in
      if depth > threshold then
        scale *. ((bid_i -. ask_i) /. (depth +. epsilon))
      else 0.0)
    bid ask
