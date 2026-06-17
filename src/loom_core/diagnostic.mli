type t = {
  loc : Location.t option;
  message : string;
  details : string list;
}

exception Error of t

val raise_error : ?loc:Location.t -> ?details:string list -> string -> 'a
val to_string : t -> string
val protect : (unit -> 'a) -> ('a, t) result

