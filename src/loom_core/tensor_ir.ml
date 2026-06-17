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

let rec scalar_expr_free_vars = function
  | SVar name -> [ name ]
  | SConstF32 _ | SConstBool _ -> []
  | SUnary (_, expr) -> scalar_expr_free_vars expr
  | SBinary (_, lhs, rhs) ->
      List.sort_uniq String.compare
        (scalar_expr_free_vars lhs @ scalar_expr_free_vars rhs)
  | SSelect (cond, then_expr, else_expr) ->
      List.sort_uniq String.compare
        (scalar_expr_free_vars cond @ scalar_expr_free_vars then_expr
       @ scalar_expr_free_vars else_expr)

let value_ref_to_string = function
  | ParamRef name -> name
  | NodeRef id -> Printf.sprintf "node_%d" id

let unary_to_string = function
  | Neg -> "neg"
  | Sqrt -> "sqrt"
  | Exp -> "exp"
  | Log -> "log"

let binary_to_string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Div -> "div"
  | Min -> "min"
  | Max -> "max"
  | CmpLt -> "cmplt"
  | CmpLe -> "cmple"
  | CmpGt -> "cmpgt"
  | CmpGe -> "cmpge"
  | CmpEq -> "cmpeq"

let rec scalar_expr_to_yojson = function
  | SVar name -> `Assoc [ ("kind", `String "var"); ("name", `String name) ]
  | SConstF32 value -> `Assoc [ ("kind", `String "f32"); ("value", `Float value) ]
  | SConstBool value ->
      `Assoc [ ("kind", `String "bool"); ("value", `Bool value) ]
  | SUnary (op, expr) ->
      `Assoc
        [ ("kind", `String "unary")
        ; ("op", `String (unary_to_string op))
        ; ("expr", scalar_expr_to_yojson expr) ]
  | SBinary (op, lhs, rhs) ->
      `Assoc
        [ ("kind", `String "binary")
        ; ("op", `String (binary_to_string op))
        ; ("lhs", scalar_expr_to_yojson lhs)
        ; ("rhs", scalar_expr_to_yojson rhs) ]
  | SSelect (cond, then_expr, else_expr) ->
      `Assoc
        [ ("kind", `String "select")
        ; ("cond", scalar_expr_to_yojson cond)
        ; ("then", scalar_expr_to_yojson then_expr)
        ; ("else", scalar_expr_to_yojson else_expr) ]

let value_ref_to_yojson value = `String (value_ref_to_string value)

let input_binding_to_yojson { name; source } =
  `Assoc [ ("name", `String name); ("source", value_ref_to_yojson source) ]

let body_metrics_to_yojson metrics =
  `Assoc
    [
      ("scalar_complexity", `Int metrics.scalar_complexity);
      ("has_branch", `Bool metrics.has_branch);
      ("has_div", `Bool metrics.has_div);
      ("branch_count", `Int metrics.branch_count);
      ("div_count", `Int metrics.div_count);
      ("repeated_subexpressions", `Int metrics.repeated_subexpressions);
      ("estimated_uses", `Int metrics.estimated_uses);
    ]

let producer_handling_hint_to_string = function
  | NoPreference -> "no-preference"
  | PreferFuse -> "prefer-fuse"
  | PreferClone -> "prefer-clone"
  | PreferMaterialize -> "prefer-materialize"

let reduction_shape_to_string = function
  | PlainReduction -> "plain"
  | MappedReduction -> "mapped"
  | WeightedMappedReduction -> "weighted-mapped"
  | BranchyMappedReduction -> "branchy-mapped"
  | RatioMappedReduction -> "ratio-mapped"

let reduction_source_to_yojson = function
  | PlainInput input ->
      `Assoc [ ("kind", `String "plain"); ("input", value_ref_to_yojson input) ]
  | MappedInput { inputs; scalar_params; body } ->
      `Assoc
        [
          ("kind", `String "mapped");
          ("inputs", `List (List.map input_binding_to_yojson inputs));
          ("scalar_params", `List (List.map (fun name -> `String name) scalar_params));
          ("body", scalar_expr_to_yojson body);
        ]

let param_to_yojson = function
  | ScalarF32 name ->
      `Assoc [ ("kind", `String "scalar-f32"); ("name", `String name) ]
  | Tensor1F32 (name, shape_symbol) ->
      `Assoc
        [ ("kind", `String "tensor1-f32"); ("name", `String name)
        ; ("shape_symbol", `String shape_symbol) ]

let result_to_yojson = function
  | TensorResult value ->
      `Assoc [ ("kind", `String "tensor"); ("value", value_ref_to_yojson value) ]
  | ScalarResult value ->
      `Assoc [ ("kind", `String "scalar"); ("value", value_ref_to_yojson value) ]

let node_to_yojson = function
  | Elementwise1D { id; shape_symbol; inputs; scalar_params; body; metrics; handling_hint } ->
      `Assoc
        [ ("kind", `String "elementwise1d")
        ; ("id", `Int id)
        ; ("shape_symbol", `String shape_symbol)
        ; ( "inputs"
          , `List
              (List.map input_binding_to_yojson inputs) )
        ; ("scalar_params", `List (List.map (fun name -> `String name) scalar_params))
        ; ("body", scalar_expr_to_yojson body)
        ; ("metrics", body_metrics_to_yojson metrics)
        ; ("handling_hint", `String (producer_handling_hint_to_string handling_hint)) ]
  | Reduce1D { id; shape_symbol; source; kind; metrics; reduction_shape; handling_hint } ->
      `Assoc
        [ ("kind", `String "reduce1d")
        ; ("id", `Int id)
        ; ("shape_symbol", `String shape_symbol)
        ; ("source", reduction_source_to_yojson source)
        ; ("reduce_kind", `String (match kind with Sum -> "sum" | MaxReduce -> "max"))
        ; ("metrics", body_metrics_to_yojson metrics)
        ; ("reduction_shape", `String (reduction_shape_to_string reduction_shape))
        ; ("handling_hint", `String (producer_handling_hint_to_string handling_hint)) ]

let program_to_yojson { entry_name; params; result; nodes } =
  `Assoc
    [ ("entry_name", `String entry_name)
    ; ("params", `List (List.map param_to_yojson params))
    ; ("result", result_to_yojson result)
    ; ("nodes", `List (List.map node_to_yojson nodes)) ]

let program_to_string program = Yojson.Safe.pretty_to_string (program_to_yojson program)
