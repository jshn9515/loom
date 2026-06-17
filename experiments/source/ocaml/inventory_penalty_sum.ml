open Loom

let[@loom.entry] inventory_penalty_sum (target : float) (pos : Tensor1.t) =
  Tensor1.reduce_sum
    (Tensor1.map
       (fun pi ->
         let d = pi -. target in
         d *. d)
       pos)
