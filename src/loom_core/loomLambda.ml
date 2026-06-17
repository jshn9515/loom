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

let prim_to_string = function
  | FAdd -> "fadd"
  | FSub -> "fsub"
  | FMul -> "fmul"
  | FDiv -> "fdiv"
  | FNeg -> "fneg"
  | FMin -> "fmin"
  | FMax -> "fmax"
  | FSqrt -> "fsqrt"
  | FExp -> "fexp"
  | FLog -> "flog"
  | FCmpLt -> "fcmplt"
  | FCmpLe -> "fcmple"
  | FCmpGt -> "fcmpgt"
  | FCmpGe -> "fcmpge"
  | FCmpEq -> "fcmpeq"
  | Select -> "select"

let tensor_prim_to_string = function
  | TensorMap -> "tensor-map"
  | TensorMap2 -> "tensor-map2"
  | TensorReduceSum -> "tensor-reduce-sum"
  | TensorReduceMax -> "tensor-reduce-max"

let json_string_assoc fields = `Assoc fields

let rec expr_to_yojson = function
  | Var (name, ty) ->
      json_string_assoc
        [ ("kind", `String "var"); ("name", `String name)
        ; ("type", `String (Loom_types.stage_type_to_string ty)) ]
  | FloatConst f ->
      json_string_assoc
        [ ("kind", `String "float"); ("value", `Float f) ]
  | BoolConst b ->
      json_string_assoc
        [ ("kind", `String "bool"); ("value", `Bool b) ]
  | Let (name, value, body) ->
      json_string_assoc
        [ ("kind", `String "let"); ("name", `String name)
        ; ("value", expr_to_yojson value); ("body", expr_to_yojson body) ]
  | If (cond, then_expr, else_expr) ->
      json_string_assoc
        [ ("kind", `String "if")
        ; ("cond", expr_to_yojson cond)
        ; ("then", expr_to_yojson then_expr)
        ; ("else", expr_to_yojson else_expr) ]
  | Lambda (params, body) ->
      json_string_assoc
        [ ("kind", `String "lambda")
        ; ( "params"
          , `List
              (List.map
                 (fun { name; ty } ->
                   json_string_assoc
                     [ ("name", `String name)
                     ; ("type", `String (Loom_types.stage_type_to_string ty)) ])
                 params) )
        ; ("body", expr_to_yojson body) ]
  | Apply (fn, args) ->
      json_string_assoc
        [ ("kind", `String "apply")
        ; ("fn", expr_to_yojson fn)
        ; ("args", `List (List.map expr_to_yojson args)) ]
  | Prim (op, args) ->
      json_string_assoc
        [ ("kind", `String "prim")
        ; ("op", `String (prim_to_string op))
        ; ("args", `List (List.map expr_to_yojson args)) ]
  | TensorPrim (kind, args) ->
      json_string_assoc
        [ ("kind", `String "tensor-prim")
        ; ("op", `String (tensor_prim_to_string kind))
        ; ("args", `List (List.map expr_to_yojson args)) ]

let entry_to_yojson { name; params; body; return_type } =
  json_string_assoc
    [ ("entry", `String name)
    ; ( "params"
      , `List
          (List.map
             (fun { name; ty } ->
               json_string_assoc
                 [ ("name", `String name)
                 ; ("type", `String (Loom_types.stage_type_to_string ty)) ])
             params) )
    ; ("return_type", `String (Loom_types.stage_type_to_string return_type))
    ; ("body", expr_to_yojson body) ]

let entry_to_string entry = Yojson.Safe.pretty_to_string (entry_to_yojson entry)
