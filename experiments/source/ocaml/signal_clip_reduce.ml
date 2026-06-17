open Loom

let[@loom.entry] signal_clip_reduce (scale : float) (epsilon : float) (clip : float)
    (bid : Tensor1.t) (ask : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map2
       (fun bid_i ask_i ->
         let base = scale *. ((bid_i -. ask_i) /. (bid_i +. ask_i +. epsilon)) in
         if base > clip then clip else if base < -.clip then -.clip else base)
       bid ask)
