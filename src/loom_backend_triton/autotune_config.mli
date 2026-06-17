type config = {
  block_size : int;
  num_warps : int;
  num_stages : int;
}

type family = {
  configs : config list;
}

type t = {
  version : int;
  bucket_upper_bounds : int list;
  elementwise : family;
  reduction : family;
  source_path : string option;
}

val default_path : string
val load : string -> t
val to_yojson : t -> Yojson.Safe.t
val to_string : t -> string
