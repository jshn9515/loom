type prim =
  | FAdd
  | FSub
  | FMul
  | FDiv
  | FNeg
  | FMin
  | FMax
  | FSqrt
  | FExp
  | FLog
  | FCmpLt
  | FCmpLe
  | FCmpGt
  | FCmpGe
  | FCmpEq
  | Select

type tensor_prim =
  | TensorMap
  | TensorMap2
  | TensorReduceSum
  | TensorReduceMax

type param = {
  name : string;
  ty : Loom_types.stage_type;
}

type expr =
  | Var of string * Loom_types.stage_type
  | FloatConst of float
  | BoolConst of bool
  | Let of string * expr * expr
  | If of expr * expr * expr
  | Lambda of param list * expr
  | Apply of expr * expr list
  | Prim of prim * expr list
  | TensorPrim of tensor_prim * expr list

type entry = {
  name : string;
  params : param list;
  body : expr;
  return_type : Loom_types.stage_type;
}

val prim_to_string : prim -> string
val tensor_prim_to_string : tensor_prim -> string
val expr_to_yojson : expr -> Yojson.Safe.t
val entry_to_yojson : entry -> Yojson.Safe.t
val entry_to_string : entry -> string
