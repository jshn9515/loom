include module type of Kernel_plan

val complexity_bucket_to_yojson : complexity_bucket -> Yojson.Safe.t
val producer_strategy_to_string : producer_strategy -> string
val pointwise_class_to_string : pointwise_class -> string
