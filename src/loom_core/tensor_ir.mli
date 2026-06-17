type scalar_unary_op =
  | Neg
  | Sqrt
  | Exp
  | Log

type scalar_binary_op =
  | Add
  | Sub
  | Mul
  | Div
  | Min
  | Max
  | CmpLt
  | CmpLe
  | CmpGt
  | CmpGe
  | CmpEq

type scalar_expr =
  | SVar of string
  | SConstF32 of float
  | SConstBool of bool
  | SUnary of scalar_unary_op * scalar_expr
  | SBinary of scalar_binary_op * scalar_expr * scalar_expr
  | SSelect of scalar_expr * scalar_expr * scalar_expr

type value_ref =
  | ParamRef of string
  | NodeRef of int

type param =
  | ScalarF32 of string
  | Tensor1F32 of string * string

type result =
  | TensorResult of value_ref
  | ScalarResult of value_ref

type reduce_kind =
  | Sum
  | MaxReduce

type input_binding = {
  name : string;
  source : value_ref;
}

type body_metrics = {
  scalar_complexity : int;
  has_branch : bool;
  has_div : bool;
  branch_count : int;
  div_count : int;
  repeated_subexpressions : int;
  estimated_uses : int;
}

type producer_handling_hint =
  | NoPreference
  | PreferFuse
  | PreferClone
  | PreferMaterialize

type reduction_shape =
  | PlainReduction
  | MappedReduction
  | WeightedMappedReduction
  | BranchyMappedReduction
  | RatioMappedReduction

type reduction_source =
  | PlainInput of value_ref
  | MappedInput of {
      inputs : input_binding list;
      scalar_params : string list;
      body : scalar_expr;
    }

type node =
  | Elementwise1D of {
      id : int;
      shape_symbol : string;
      inputs : input_binding list;
      scalar_params : string list;
      body : scalar_expr;
      metrics : body_metrics;
      handling_hint : producer_handling_hint;
    }
  | Reduce1D of {
      id : int;
      shape_symbol : string;
      source : reduction_source;
      kind : reduce_kind;
      metrics : body_metrics;
      reduction_shape : reduction_shape;
      handling_hint : producer_handling_hint;
    }

type program = {
  entry_name : string;
  params : param list;
  result : result;
  nodes : node list;
}

val scalar_expr_free_vars : scalar_expr -> string list
val scalar_expr_to_yojson : scalar_expr -> Yojson.Safe.t
val value_ref_to_string : value_ref -> string
val body_metrics_to_yojson : body_metrics -> Yojson.Safe.t
val producer_handling_hint_to_string : producer_handling_hint -> string
val reduction_shape_to_string : reduction_shape -> string
val program_to_yojson : program -> Yojson.Safe.t
val program_to_string : program -> string
