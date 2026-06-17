open Loom

let[@loom.entry] scaled_book_signal (scale : float) (epsilon : float) (bid : Tensor1.t)
    (ask : Tensor1.t) =
  Tensor1.map2
    (fun bid_i ask_i -> scale *. ((bid_i -. ask_i) /. (bid_i +. ask_i +. epsilon)))
    bid ask
