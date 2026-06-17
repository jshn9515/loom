type t = unit

let staged name =
  invalid_arg
    (Printf.sprintf
       "Loom.Tensor1.%s is a staged primitive; compile this file with loomc" name)

let map _ _ = staged "map"
let map2 _ _ _ = staged "map2"
let reduce_sum _ = staged "reduce_sum"
let reduce_max _ = staged "reduce_max"
