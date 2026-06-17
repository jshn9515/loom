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

type pattern =
  | PVar of string * Loom_types.stage_type
  | PTuple of pattern list

type expr =
  | Var of string * Loom_types.stage_type
  | FloatConst of float
  | BoolConst of bool
  | UnitConst
  | Tuple of expr list
  | Let of pattern * expr * expr
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

let json_assoc fields = `Assoc fields

let expect_assoc = function
  | `Assoc fields -> fields
  | _ -> Diagnostic.raise_error "expected JSON object while decoding FrontIR"

let expect_string field = function
  | `String value -> value
  | _ -> Diagnostic.raise_error (Printf.sprintf "expected string for FrontIR field %s" field)

let expect_list field = function
  | `List items -> items
  | _ -> Diagnostic.raise_error (Printf.sprintf "expected list for FrontIR field %s" field)

let field fields name =
  match List.assoc_opt name fields with
  | Some value -> value
  | None -> Diagnostic.raise_error (Printf.sprintf "missing FrontIR field %s" name)

let param_to_yojson { name; ty } =
  json_assoc
    [ ("name", `String name)
    ; ("type", `String (Loom_types.stage_type_to_string ty)) ]

let param_of_yojson json =
  let fields = expect_assoc json in
  let name = field fields "name" |> expect_string "name" in
  let ty =
    field fields "type"
    |> expect_string "type"
    |> Loom_types.stage_type_of_string
  in
  match ty with
  | Some ty -> { name; ty }
  | None -> Diagnostic.raise_error "unknown FrontIR stage type"

let rec pattern_to_yojson = function
  | PVar (name, ty) ->
      json_assoc
        [ ("kind", `String "var")
        ; ("name", `String name)
        ; ("type", `String (Loom_types.stage_type_to_string ty)) ]
  | PTuple items ->
      json_assoc
        [ ("kind", `String "tuple")
        ; ("items", `List (List.map pattern_to_yojson items)) ]

let rec pattern_of_yojson json =
  let fields = expect_assoc json in
  match field fields "kind" |> expect_string "kind" with
  | "var" ->
      let name = field fields "name" |> expect_string "name" in
      let ty =
        field fields "type"
        |> expect_string "type"
        |> Loom_types.stage_type_of_string
      in
      begin
        match ty with
        | Some ty -> PVar (name, ty)
        | None -> Diagnostic.raise_error "unknown FrontIR pattern type"
      end
  | "tuple" ->
      PTuple
        (field fields "items"
        |> expect_list "items"
        |> List.map pattern_of_yojson)
  | kind ->
      Diagnostic.raise_error
        (Printf.sprintf "unknown FrontIR pattern kind %s" kind)

let rec expr_to_yojson = function
  | Var (name, ty) ->
      json_assoc
        [ ("kind", `String "var")
        ; ("name", `String name)
        ; ("type", `String (Loom_types.stage_type_to_string ty)) ]
  | FloatConst value -> json_assoc [ ("kind", `String "float"); ("value", `Float value) ]
  | BoolConst value -> json_assoc [ ("kind", `String "bool"); ("value", `Bool value) ]
  | UnitConst -> json_assoc [ ("kind", `String "unit") ]
  | Tuple items ->
      json_assoc
        [ ("kind", `String "tuple")
        ; ("items", `List (List.map expr_to_yojson items)) ]
  | Let (pattern, value, body) ->
      json_assoc
        [ ("kind", `String "let")
        ; ("pattern", pattern_to_yojson pattern)
        ; ("value", expr_to_yojson value)
        ; ("body", expr_to_yojson body) ]
  | If (cond, then_expr, else_expr) ->
      json_assoc
        [ ("kind", `String "if")
        ; ("cond", expr_to_yojson cond)
        ; ("then", expr_to_yojson then_expr)
        ; ("else", expr_to_yojson else_expr) ]
  | Lambda (params, body) ->
      json_assoc
        [ ("kind", `String "lambda")
        ; ("params", `List (List.map param_to_yojson params))
        ; ("body", expr_to_yojson body) ]
  | Apply (fn, args) ->
      json_assoc
        [ ("kind", `String "apply")
        ; ("fn", expr_to_yojson fn)
        ; ("args", `List (List.map expr_to_yojson args)) ]
  | Prim (op, args) ->
      json_assoc
        [ ("kind", `String "prim")
        ; ("op", `String (prim_to_string op))
        ; ("args", `List (List.map expr_to_yojson args)) ]
  | TensorPrim (op, args) ->
      json_assoc
        [ ("kind", `String "tensor-prim")
        ; ("op", `String (tensor_prim_to_string op))
        ; ("args", `List (List.map expr_to_yojson args)) ]

let prim_of_string = function
  | "fadd" -> FAdd
  | "fsub" -> FSub
  | "fmul" -> FMul
  | "fdiv" -> FDiv
  | "fneg" -> FNeg
  | "fmin" -> FMin
  | "fmax" -> FMax
  | "fsqrt" -> FSqrt
  | "fexp" -> FExp
  | "flog" -> FLog
  | "fcmplt" -> FCmpLt
  | "fcmple" -> FCmpLe
  | "fcmpgt" -> FCmpGt
  | "fcmpge" -> FCmpGe
  | "fcmpeq" -> FCmpEq
  | "select" -> Select
  | value -> Diagnostic.raise_error (Printf.sprintf "unknown FrontIR primitive %s" value)

let tensor_prim_of_string = function
  | "tensor-map" -> TensorMap
  | "tensor-map2" -> TensorMap2
  | "tensor-reduce-sum" -> TensorReduceSum
  | "tensor-reduce-max" -> TensorReduceMax
  | value -> Diagnostic.raise_error (Printf.sprintf "unknown FrontIR tensor primitive %s" value)

let rec expr_of_yojson json =
  let fields = expect_assoc json in
  match field fields "kind" |> expect_string "kind" with
  | "var" ->
      let name = field fields "name" |> expect_string "name" in
      let ty =
        field fields "type"
        |> expect_string "type"
        |> Loom_types.stage_type_of_string
      in
      begin
        match ty with
        | Some ty -> Var (name, ty)
        | None -> Diagnostic.raise_error "unknown FrontIR variable type"
      end
  | "float" -> (
      match field fields "value" with
      | `Float value -> FloatConst value
      | `Int value -> FloatConst (float_of_int value)
      | _ -> Diagnostic.raise_error "expected numeric FrontIR float value" )
  | "bool" -> (
      match field fields "value" with
      | `Bool value -> BoolConst value
      | _ -> Diagnostic.raise_error "expected bool FrontIR value" )
  | "unit" -> UnitConst
  | "tuple" ->
      Tuple
        (field fields "items"
        |> expect_list "items"
        |> List.map expr_of_yojson)
  | "let" ->
      Let
        ( pattern_of_yojson (field fields "pattern")
        , expr_of_yojson (field fields "value")
        , expr_of_yojson (field fields "body") )
  | "if" ->
      If
        ( expr_of_yojson (field fields "cond")
        , expr_of_yojson (field fields "then")
        , expr_of_yojson (field fields "else") )
  | "lambda" ->
      Lambda
        ( field fields "params"
          |> expect_list "params"
          |> List.map param_of_yojson
        , expr_of_yojson (field fields "body") )
  | "apply" ->
      Apply
        ( expr_of_yojson (field fields "fn")
        , field fields "args" |> expect_list "args" |> List.map expr_of_yojson )
  | "prim" ->
      Prim
        ( field fields "op" |> expect_string "op" |> prim_of_string
        , field fields "args" |> expect_list "args" |> List.map expr_of_yojson )
  | "tensor-prim" ->
      TensorPrim
        ( field fields "op" |> expect_string "op" |> tensor_prim_of_string
        , field fields "args" |> expect_list "args" |> List.map expr_of_yojson )
  | kind -> Diagnostic.raise_error (Printf.sprintf "unknown FrontIR expression kind %s" kind)

let entry_to_yojson { name; params; body; return_type } =
  json_assoc
    [ ("entry", `String name)
    ; ("params", `List (List.map param_to_yojson params))
    ; ("return_type", `String (Loom_types.stage_type_to_string return_type))
    ; ("body", expr_to_yojson body) ]

let entry_to_string entry = Yojson.Safe.pretty_to_string (entry_to_yojson entry)

let entry_of_yojson json =
  let fields = expect_assoc json in
  let name = field fields "entry" |> expect_string "entry" in
  let params = field fields "params" |> expect_list "params" |> List.map param_of_yojson in
  let return_type =
    field fields "return_type"
    |> expect_string "return_type"
    |> Loom_types.stage_type_of_string
  in
  match return_type with
  | Some return_type ->
      { name; params; body = expr_of_yojson (field fields "body"); return_type }
  | None -> Diagnostic.raise_error "unknown FrontIR return type"

let entry_of_string text = Yojson.Safe.from_string text |> entry_of_yojson
