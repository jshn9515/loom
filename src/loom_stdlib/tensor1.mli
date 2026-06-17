type t

val map : (float -> float) -> t -> t
val map2 : (float -> float -> float) -> t -> t -> t
val reduce_sum : t -> float
val reduce_max : t -> float

