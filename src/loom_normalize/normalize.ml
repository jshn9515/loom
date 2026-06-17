open LoomLambda

module String_map = Normalize_scalar.String_map

type env = binding String_map.t

and binding =
  | ValueBinding of LoomLambda.expr
  | TupleBinding of binding list
  | FunctionBinding of Front_ir.param list * Front_ir.expr * env

type normalized_value = NExpr of LoomLambda.expr | NTuple of binding list

type bind_result =
  | Done of env * Front_ir.expr list
  | Partial of env * Front_ir.param list

let prim_of_front = function
  | Front_ir.FAdd -> LoomLambda.FAdd
  | FSub -> FSub
  | FMul -> FMul
  | FDiv -> FDiv
  | FNeg -> FNeg
  | FMin -> FMin
  | FMax -> FMax
  | FSqrt -> FSqrt
  | FExp -> FExp
  | FLog -> FLog
  | FCmpLt -> FCmpLt
  | FCmpLe -> FCmpLe
  | FCmpGt -> FCmpGt
  | FCmpGe -> FCmpGe
  | FCmpEq -> FCmpEq
  | Select -> Select

let tensor_prim_of_front = function
  | Front_ir.TensorMap -> LoomLambda.TensorMap
  | TensorMap2 -> TensorMap2
  | TensorReduceSum -> TensorReduceSum
  | TensorReduceMax -> TensorReduceMax

let rec binding_to_expr = function
  | ValueBinding expr -> expr
  | TupleBinding _ ->
      Diagnostic.raise_error
        "tuple values must be destructured before LoomLambda normalization"
  | FunctionBinding (params, body, closure) ->
      let params =
        List.map
          (fun (param : Front_ir.param) ->
            { LoomLambda.name = param.name; ty = param.ty })
          params
      in
      let closure =
        List.fold_left
          (fun acc (param : LoomLambda.param) ->
            String_map.add param.name
              (ValueBinding (LoomLambda.Var (param.name, param.ty)))
              acc)
          closure params
      in
      LoomLambda.Lambda (params, normalize_expr closure body)

and normalize_value env = function
  | Front_ir.Tuple items ->
      NTuple
        (List.map
           (fun item ->
             match normalize_value env item with
             | NExpr expr -> ValueBinding expr
             | NTuple items -> TupleBinding items)
           items)
  | Front_ir.Var (name, ty) -> (
      match String_map.find_opt name env with
      | Some (TupleBinding items) -> NTuple items
      | Some binding -> NExpr (binding_to_expr binding)
      | None -> NExpr (LoomLambda.Var (name, ty)))
  | expr -> NExpr (normalize_expr env expr)

and bind_pattern env pattern binding =
  match (pattern, binding) with
  | Front_ir.PVar ("_", _), _ -> env
  | Front_ir.PVar (name, _), _ -> String_map.add name binding env
  | Front_ir.PTuple patterns, TupleBinding items ->
      if List.length patterns <> List.length items then
        Diagnostic.raise_error
          "tuple pattern arity mismatch during normalization";
      List.fold_left2 bind_pattern env patterns items
  | Front_ir.PTuple _, _ ->
      Diagnostic.raise_error
        "tuple destructuring requires a tuple-producing expression during \
         normalization"

and bind_params (env : env) (params : Front_ir.param list)
    (args : Front_ir.expr list) : bind_result =
  match (params, args) with
  | [], rest -> Done (env, rest)
  | param :: params, arg :: args ->
      let binding =
        match normalize_value env arg with
        | NExpr expr -> ValueBinding expr
        | NTuple items -> TupleBinding items
      in
      bind_params (String_map.add param.Front_ir.name binding env) params args
  | remaining, [] -> Partial (env, remaining)

and normalize_apply env fn_expr args =
  LoomLambda.Apply (fn_expr, List.map (normalize_expr env) args)

and apply_function closure params body args =
  match bind_params closure params args with
  | Done (env, []) -> normalize_expr env body
  | Done (env, rest) -> normalize_apply env (normalize_expr env body) rest
  | Partial (env, remaining) ->
      let remaining =
        List.map
          (fun (param : Front_ir.param) ->
            { LoomLambda.name = param.name; ty = param.ty })
          remaining
      in
      LoomLambda.Lambda (remaining, normalize_expr env body)

and normalize_expr env = function
  | Front_ir.Var (name, ty) -> (
      match String_map.find_opt name env with
      | Some binding -> binding_to_expr binding
      | None -> LoomLambda.Var (name, ty))
  | Front_ir.FloatConst value -> LoomLambda.FloatConst value
  | Front_ir.BoolConst value -> LoomLambda.BoolConst value
  | Front_ir.UnitConst -> LoomLambda.Var ("()", Loom_types.LLUnit)
  | Front_ir.Tuple _ ->
      Diagnostic.raise_error
        "tuple expressions must be eliminated during normalization"
  | Front_ir.Let (pattern, value, body) -> (
      match (pattern, value) with
      | Front_ir.PVar (name, _), Front_ir.Lambda (params, lambda_body) ->
          let env =
            String_map.add name (FunctionBinding (params, lambda_body, env)) env
          in
          normalize_expr env body
      | _ -> (
          match (pattern, normalize_value env value) with
          | Front_ir.PVar (name, ty), NExpr expr ->
              let body_env =
                String_map.add name
                  (ValueBinding (LoomLambda.Var (name, ty)))
                  env
              in
              LoomLambda.Let (name, expr, normalize_expr body_env body)
          | _, normalized ->
              let binding =
                match normalized with
                | NExpr expr -> ValueBinding expr
                | NTuple items -> TupleBinding items
              in
              normalize_expr (bind_pattern env pattern binding) body))
  | Front_ir.If (cond, then_expr, else_expr) ->
      LoomLambda.If
        ( normalize_expr env cond,
          normalize_expr env then_expr,
          normalize_expr env else_expr )
  | Front_ir.Lambda (params, body) ->
      let lambda_params =
        List.map
          (fun (param : Front_ir.param) ->
            { LoomLambda.name = param.name; ty = param.ty })
          params
      in
      let env =
        List.fold_left
          (fun acc (param : Front_ir.param) ->
            String_map.add param.name
              (ValueBinding (LoomLambda.Var (param.name, param.ty)))
              acc)
          env params
      in
      LoomLambda.Lambda (lambda_params, normalize_expr env body)
  | Front_ir.Apply (Front_ir.Var (name, _), args) -> (
      match String_map.find_opt name env with
      | Some (FunctionBinding (params, body, closure)) ->
          apply_function closure params body args
      | Some binding -> normalize_apply env (binding_to_expr binding) args
      | None ->
          normalize_apply env (LoomLambda.Var (name, Loom_types.LLUnit)) args)
  | Front_ir.Apply (Front_ir.Lambda (params, body), args) ->
      apply_function env params body args
  | Front_ir.Apply (fn, args) ->
      normalize_apply env (normalize_expr env fn) args
  | Front_ir.Prim (prim, args) ->
      LoomLambda.Prim (prim_of_front prim, List.map (normalize_expr env) args)
  | Front_ir.TensorPrim (kind, args) ->
      LoomLambda.TensorPrim
        (tensor_prim_of_front kind, List.map (normalize_expr env) args)

let apply_optimizations optimizations entry =
  let body = entry.LoomLambda.body in
  let body =
    if Optimizations.enabled optimizations Optimizations.ScalarConstFold then
      Normalize_scalar.const_fold String_map.empty body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.IfSimplify then
      Normalize_scalar.if_simplify body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.ArithReassociate then
      Normalize_scalar.arith_reassociate body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.LetFloat then
      Normalize_scalar.let_float body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.BranchHoist then
      Normalize_scalar.branch_hoist body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.LambdaInlineSmall then
      Normalize_scalar.lambda_inline_small optimizations.config.small_inline_nodes
        body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.ScalarCse then
      Normalize_scalar.scalar_cse String_map.empty body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.NormalizedDce then
      Normalize_scalar.dce body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.ScalarHoist then
      Normalize_scalar.scalar_hoist body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.ScalarConstFold then
      Normalize_scalar.const_fold String_map.empty body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.IfSimplify then
      Normalize_scalar.if_simplify body
    else body
  in
  let body =
    if Optimizations.enabled optimizations Optimizations.NormalizedDce then
      Normalize_scalar.dce body
    else body
  in
  { entry with body }

let entry_of_front_ir ?(optimizations = Optimizations.none)
    (entry : Front_ir.entry) =
  let env =
    List.fold_left
      (fun acc (param : Front_ir.param) ->
        String_map.add param.name
          (ValueBinding (LoomLambda.Var (param.name, param.ty)))
          acc)
      String_map.empty entry.params
  in
  {
    LoomLambda.name = entry.name;
    params =
      List.map
        (fun (param : Front_ir.param) ->
          { LoomLambda.name = param.name; ty = param.ty })
        entry.params;
    body = normalize_expr env entry.body;
    return_type = entry.return_type;
  }
  |> apply_optimizations optimizations
