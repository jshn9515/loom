open Typedtree

let rec pattern_name (pat : Typedtree.pattern) =
  match pat.pat_desc with
  | Tpat_any -> "_"
  | Tpat_var (ident, _, _) -> Ident.name ident
  | Tpat_alias (inner, ident, _, _, _) ->
      let inner_name = pattern_name inner in
      if inner_name = "_" then Ident.name ident else inner_name
  | _ -> Diagnostic.raise_error ~loc:pat.pat_loc "only simple variable patterns are supported here"

let rec flatten_function (expr : Typedtree.expression) =
  match expr.exp_desc with
  | Texp_function (params, Tfunction_body body) ->
      let params', body' = flatten_function body in
      (params @ params', body')
  | Texp_function (_, Tfunction_cases _) ->
      Diagnostic.raise_error ~loc:expr.exp_loc "pattern-matching functions are not supported"
  | _ -> ([], expr)

let classify_type ids loc ty =
  match Loom_types.classify_type ~tensor_type:ids.Ocaml_entry_scan.tensor_type_path ty with
  | Some ty -> ty
  | None -> Diagnostic.raise_error ~loc "unsupported staged type"

let rec pattern_of_typed_pattern ids (pat : Typedtree.pattern) =
  let ty = classify_type ids pat.pat_loc pat.pat_type in
  match pat.pat_desc with
  | Tpat_any -> Front_ir.PVar ("_", ty)
  | Tpat_var (ident, _, _) -> Front_ir.PVar (Ident.name ident, ty)
  | Tpat_alias (inner, ident, _, _, _) -> (
      match pattern_of_typed_pattern ids inner with
      | Front_ir.PVar ("_", _) -> Front_ir.PVar (Ident.name ident, ty)
      | other -> other )
  | Tpat_tuple pats ->
      Front_ir.PTuple (List.map (fun (_, pat) -> pattern_of_typed_pattern ids pat) pats)
  | _ ->
      Diagnostic.raise_error ~loc:pat.pat_loc
        "unsupported pattern in staged region"

let param_of_function_param ids (param : Typedtree.function_param) =
  match param.fp_kind with
  | Tparam_optional_default _ ->
      Diagnostic.raise_error ~loc:param.fp_loc "optional parameters are not supported"
  | Tparam_pat pat -> (
      match pattern_of_typed_pattern ids pat with
      | Front_ir.PVar (name, ty) -> { Front_ir.name; ty }
      | Front_ir.PTuple _ ->
          Diagnostic.raise_error ~loc:pat.pat_loc
            "tuple patterns are not supported in lambda parameters" )

let prim_of_ident name arity =
  match (name, arity) with
  | ("+.", 2) -> Some Front_ir.FAdd
  | ("-.", 2) -> Some Front_ir.FSub
  | ("*.", 2) -> Some Front_ir.FMul
  | ("/.", 2) -> Some Front_ir.FDiv
  | ("~-.", 1) -> Some Front_ir.FNeg
  | ("sqrt", 1) -> Some Front_ir.FSqrt
  | ("exp", 1) -> Some Front_ir.FExp
  | ("log", 1) -> Some Front_ir.FLog
  | ("min", 2) -> Some Front_ir.FMin
  | ("max", 2) -> Some Front_ir.FMax
  | ("<", 2) -> Some Front_ir.FCmpLt
  | ("<=", 2) -> Some Front_ir.FCmpLe
  | (">", 2) -> Some Front_ir.FCmpGt
  | (">=", 2) -> Some Front_ir.FCmpGe
  | ("=", 2) -> Some Front_ir.FCmpEq
  | _ -> None

let tensor_prim_of_path ids path name =
  if Path.same path ids.Ocaml_entry_scan.map_path then Some Front_ir.TensorMap
  else if Path.same path ids.map2_path then Some Front_ir.TensorMap2
  else if Path.same path ids.reduce_sum_path then Some Front_ir.TensorReduceSum
  else if Path.same path ids.reduce_max_path then Some Front_ir.TensorReduceMax
  else
    match name with
    | "map" -> Some Front_ir.TensorMap
    | "map2" -> Some Front_ir.TensorMap2
    | "reduce_sum" -> Some Front_ir.TensorReduceSum
    | "reduce_max" -> Some Front_ir.TensorReduceMax
    | _ -> None

let expr_of_constant loc = function
  | Asttypes.Const_float text -> Front_ir.FloatConst (float_of_string text)
  | _ -> Diagnostic.raise_error ~loc "unsupported constant in staged region"

let rec import_expr ids (expr : Typedtree.expression) =
  match expr.exp_desc with
  | Texp_ident (_, lid, _) ->
      let ty = classify_type ids expr.exp_loc expr.exp_type in
      Front_ir.Var (Longident.last lid.txt, ty)
  | Texp_constant constant -> expr_of_constant expr.exp_loc constant
  | Texp_construct ({ txt = Longident.Lident "true"; _ }, _, []) -> Front_ir.BoolConst true
  | Texp_construct ({ txt = Longident.Lident "false"; _ }, _, []) -> Front_ir.BoolConst false
  | Texp_construct ({ txt = Longident.Lident "()"; _ }, _, []) -> Front_ir.UnitConst
  | Texp_let (Recursive, _, _) ->
      Diagnostic.raise_error ~loc:expr.exp_loc "recursive lets are not supported in staged regions"
  | Texp_let (Nonrecursive, bindings, body) ->
      List.fold_right
        (fun binding acc ->
          Front_ir.Let
            (pattern_of_typed_pattern ids binding.vb_pat, import_expr ids binding.vb_expr, acc))
        bindings (import_expr ids body)
  | Texp_ifthenelse (cond, then_expr, Some else_expr) ->
      Front_ir.If (import_expr ids cond, import_expr ids then_expr, import_expr ids else_expr)
  | Texp_ifthenelse (_, _, None) ->
      Diagnostic.raise_error ~loc:expr.exp_loc "if expressions in staged regions require an else branch"
  | Texp_function (params, Tfunction_body body) ->
      Front_ir.Lambda
        (List.map (param_of_function_param ids) params, import_expr ids body)
  | Texp_function (_, Tfunction_cases _) ->
      Diagnostic.raise_error ~loc:expr.exp_loc "pattern-matching lambdas are not supported"
  | Texp_apply ({ exp_desc = Texp_ident (path, lid, _); exp_type = fn_type; _ }, args) -> (
      let ident_name = Longident.last lid.txt in
      let args =
        List.map
          (function
            | _, Arg arg -> import_expr ids arg
            | _, Omitted _ ->
                Diagnostic.raise_error ~loc:expr.exp_loc "omitted arguments are not supported")
          args
      in
      match tensor_prim_of_path ids path ident_name with
      | Some kind -> Front_ir.TensorPrim (kind, args)
      | None -> (
          match prim_of_ident ident_name (List.length args) with
          | Some prim -> Front_ir.Prim (prim, args)
          | None ->
              Front_ir.Apply
                (Front_ir.Var (Path.name path, classify_type ids expr.exp_loc fn_type), args) ) )
  | Texp_apply (fn, args) ->
      let args =
        List.map
          (function
            | _, Arg arg -> import_expr ids arg
            | _, Omitted _ ->
                Diagnostic.raise_error ~loc:expr.exp_loc "omitted arguments are not supported")
          args
      in
      Front_ir.Apply (import_expr ids fn, args)
  | Texp_tuple exprs ->
      Front_ir.Tuple (List.map (fun (_, expr) -> import_expr ids expr) exprs)
  | _ ->
      Diagnostic.raise_error ~loc:expr.exp_loc
        "unsupported staged expression form"

let import_entry entry =
  let ids = entry.Ocaml_entry_scan.owner.primitive_ids in
  let params, body = flatten_function entry.binding.vb_expr in
  let params = List.map (param_of_function_param ids) params in
  let return_type = classify_type ids body.exp_loc body.exp_type in
  { Front_ir.name = entry.name
  ; params
  ; body = import_expr ids body
  ; return_type }
