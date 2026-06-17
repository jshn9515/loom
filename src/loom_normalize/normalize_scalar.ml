open LoomLambda

module String_map = Map.Make (String)
module String_set = Set.Make (String)

let rec free_vars = function
  | LoomLambda.Var (name, _) -> String_set.singleton name
  | FloatConst _ | BoolConst _ -> String_set.empty
  | Let (name, value, body) ->
      String_set.union (free_vars value)
        (String_set.remove name (free_vars body))
  | If (cond, then_expr, else_expr) ->
      String_set.union (free_vars cond)
        (String_set.union (free_vars then_expr) (free_vars else_expr))
  | Lambda (params, body) ->
      List.fold_left
        (fun acc (param : LoomLambda.param) -> String_set.remove param.name acc)
        (free_vars body) params
  | Apply (fn, args) ->
      List.fold_left
        (fun acc expr -> String_set.union acc (free_vars expr))
        (free_vars fn) args
  | TensorPrim (_, args) | Prim (_, args) ->
      List.fold_left
        (fun acc expr -> String_set.union acc (free_vars expr))
        String_set.empty args

let is_float = function LoomLambda.FloatConst value -> Some value | _ -> None
let is_bool = function LoomLambda.BoolConst value -> Some value | _ -> None

let eval_unary prim arg =
  match (prim, is_float arg) with
  | FNeg, Some value -> Some (FloatConst (-.value))
  | FSqrt, Some value -> Some (FloatConst (sqrt value))
  | FExp, Some value -> Some (FloatConst (exp value))
  | FLog, Some value -> Some (FloatConst (log value))
  | _ -> None

let eval_binary prim lhs rhs =
  match (prim, is_float lhs, is_float rhs, is_bool lhs, is_bool rhs) with
  | FAdd, Some lhs, Some rhs, _, _ -> Some (FloatConst (lhs +. rhs))
  | FSub, Some lhs, Some rhs, _, _ -> Some (FloatConst (lhs -. rhs))
  | FMul, Some lhs, Some rhs, _, _ -> Some (FloatConst (lhs *. rhs))
  | FDiv, Some lhs, Some rhs, _, _ -> Some (FloatConst (lhs /. rhs))
  | FMin, Some lhs, Some rhs, _, _ -> Some (FloatConst (min lhs rhs))
  | FMax, Some lhs, Some rhs, _, _ -> Some (FloatConst (max lhs rhs))
  | FCmpLt, Some lhs, Some rhs, _, _ -> Some (BoolConst (lhs < rhs))
  | FCmpLe, Some lhs, Some rhs, _, _ -> Some (BoolConst (lhs <= rhs))
  | FCmpGt, Some lhs, Some rhs, _, _ -> Some (BoolConst (lhs > rhs))
  | FCmpGe, Some lhs, Some rhs, _, _ -> Some (BoolConst (lhs >= rhs))
  | FCmpEq, Some lhs, Some rhs, _, _ -> Some (BoolConst (lhs = rhs))
  | FCmpEq, _, _, Some lhs, Some rhs -> Some (BoolConst (lhs = rhs))
  | _ -> None

let eval_select cond then_expr else_expr =
  match is_bool cond with
  | Some true -> Some then_expr
  | Some false -> Some else_expr
  | None -> None

let is_scalar_only_expr expr =
  let rec loop = function
    | LoomLambda.Var (_, ty) -> (
        match ty with Loom_types.LLFloat | LLBool -> true | _ -> false)
    | FloatConst _ | BoolConst _ -> true
    | Let (_, value, body) -> loop value && loop body
    | If (cond, then_expr, else_expr) ->
        loop cond && loop then_expr && loop else_expr
    | Lambda _ | Apply _ | TensorPrim _ -> false
    | Prim (_, args) -> List.for_all loop args
  in
  loop expr

let rec const_fold env = function
  | LoomLambda.Var (name, ty) -> (
      match String_map.find_opt name env with
      | Some expr -> expr
      | None -> Var (name, ty))
  | FloatConst _ as expr -> expr
  | BoolConst _ as expr -> expr
  | Let (name, value, body) ->
      let value = const_fold env value in
      let env =
        match value with
        | FloatConst _ | BoolConst _ -> String_map.add name value env
        | _ -> String_map.remove name env
      in
      Let (name, value, const_fold env body)
  | If (cond, then_expr, else_expr) -> (
      let cond = const_fold env cond in
      let then_expr = const_fold env then_expr in
      let else_expr = const_fold env else_expr in
      match is_bool cond with
      | Some true -> then_expr
      | Some false -> else_expr
      | None -> If (cond, then_expr, else_expr))
  | Lambda (params, body) ->
      let env =
        List.fold_left
          (fun acc (param : LoomLambda.param) ->
            String_map.remove param.name acc)
          env params
      in
      Lambda (params, const_fold env body)
  | Apply (fn, args) -> Apply (const_fold env fn, List.map (const_fold env) args)
  | Prim (prim, args) ->
      let args = List.map (const_fold env) args in
      begin match (prim, args) with
      | Select, [ cond; then_expr; else_expr ] -> (
          match eval_select cond then_expr else_expr with
          | Some expr -> expr
          | None -> Prim (prim, args))
      | (FNeg | FSqrt | FExp | FLog), [ arg ] -> (
          match eval_unary prim arg with
          | Some expr -> expr
          | None -> Prim (prim, args))
      | _, [ lhs; rhs ] -> (
          match eval_binary prim lhs rhs with
          | Some expr -> expr
          | None -> Prim (prim, args))
      | _ -> Prim (prim, args)
      end
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, const_fold env body) :: List.map (const_fold env) tl
        | _ -> List.map (const_fold env) args
      in
      TensorPrim (kind, args)

let rec expr_size = function
  | LoomLambda.Var _ | FloatConst _ | BoolConst _ -> 1
  | Let (_, value, body) -> 1 + expr_size value + expr_size body
  | If (cond, then_expr, else_expr) ->
      1 + expr_size cond + expr_size then_expr + expr_size else_expr
  | Lambda (_, body) -> 1 + expr_size body
  | Apply (fn, args) ->
      1 + expr_size fn
      + List.fold_left (fun acc arg -> acc + expr_size arg) 0 args
  | Prim (_, args) | TensorPrim (_, args) ->
      1 + List.fold_left (fun acc arg -> acc + expr_size arg) 0 args

let rec if_simplify = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) -> Let (name, if_simplify value, if_simplify body)
  | If (cond, then_expr, else_expr) -> (
      let cond = if_simplify cond in
      let then_expr = if_simplify then_expr in
      let else_expr = if_simplify else_expr in
      match (cond, then_expr, else_expr) with
      | BoolConst true, _, _ -> then_expr
      | BoolConst false, _, _ -> else_expr
      | _, _, _ when then_expr = else_expr -> then_expr
      | _ -> If (cond, then_expr, else_expr))
  | Lambda (params, body) -> Lambda (params, if_simplify body)
  | Apply (fn, args) -> Apply (if_simplify fn, List.map if_simplify args)
  | Prim (Select, [ cond; then_expr; else_expr ]) ->
      let cond = if_simplify cond in
      let then_expr = if_simplify then_expr in
      let else_expr = if_simplify else_expr in
      if then_expr = else_expr then then_expr
      else Prim (Select, [ cond; then_expr; else_expr ])
  | Prim (prim, args) -> Prim (prim, List.map if_simplify args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, if_simplify body) :: List.map if_simplify tl
        | _ -> List.map if_simplify args
      in
      TensorPrim (kind, args)

let rec let_float = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) -> Let (name, let_float value, let_float body)
  | If
      ( cond,
        Let (then_name, then_value, then_body),
        Let (else_name, else_value, else_body) )
    when String.equal then_name else_name
         && then_value = else_value
         && is_scalar_only_expr then_value ->
      Let
        ( then_name,
          let_float then_value,
          If (let_float cond, let_float then_body, let_float else_body) )
  | If (cond, then_expr, else_expr) ->
      If (let_float cond, let_float then_expr, let_float else_expr)
  | Lambda (params, body) -> Lambda (params, let_float body)
  | Apply (fn, args) -> Apply (let_float fn, List.map let_float args)
  | Prim (prim, args) -> Prim (prim, List.map let_float args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, let_float body) :: List.map let_float tl
        | _ -> List.map let_float args
      in
      TensorPrim (kind, args)

let rec lambda_inline_small max_nodes = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) ->
      Let
        ( name,
          lambda_inline_small max_nodes value,
          lambda_inline_small max_nodes body )
  | If (cond, then_expr, else_expr) ->
      If
        ( lambda_inline_small max_nodes cond,
          lambda_inline_small max_nodes then_expr,
          lambda_inline_small max_nodes else_expr )
  | Lambda (params, body) -> Lambda (params, lambda_inline_small max_nodes body)
  | Apply (Lambda (params, body), args)
    when List.length params = List.length args && expr_size body <= max_nodes ->
      List.fold_right2
        (fun (param : LoomLambda.param) arg acc ->
          LoomLambda.Let (param.name, lambda_inline_small max_nodes arg, acc))
        params args
        (lambda_inline_small max_nodes body)
  | Apply (fn, args) ->
      Apply
        ( lambda_inline_small max_nodes fn,
          List.map (lambda_inline_small max_nodes) args )
  | Prim (prim, args) ->
      Prim (prim, List.map (lambda_inline_small max_nodes) args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, lambda_inline_small max_nodes body)
            :: List.map (lambda_inline_small max_nodes) tl
        | _ -> List.map (lambda_inline_small max_nodes) args
      in
      TensorPrim (kind, args)

let rec expr_fingerprint = function
  | LoomLambda.Var (name, _) -> "v:" ^ name
  | FloatConst value -> Printf.sprintf "f:%0.12g" value
  | BoolConst value -> Printf.sprintf "b:%b" value
  | Let (name, value, body) ->
      Printf.sprintf "l:%s:%s:%s" name (expr_fingerprint value)
        (expr_fingerprint body)
  | If (cond, then_expr, else_expr) ->
      Printf.sprintf "i:%s:%s:%s" (expr_fingerprint cond)
        (expr_fingerprint then_expr) (expr_fingerprint else_expr)
  | Lambda (params, body) ->
      let params =
        params
        |> List.map (fun (param : LoomLambda.param) -> param.name)
        |> String.concat ","
      in
      Printf.sprintf "fn:%s:%s" params (expr_fingerprint body)
  | Apply (fn, args) ->
      Printf.sprintf "app:%s:%s" (expr_fingerprint fn)
        (args |> List.map expr_fingerprint |> String.concat ",")
  | Prim (prim, args) ->
      Printf.sprintf "p:%s:%s" (LoomLambda.prim_to_string prim)
        (args |> List.map expr_fingerprint |> String.concat ",")
  | TensorPrim (kind, args) ->
      Printf.sprintf "t:%s:%s" (LoomLambda.tensor_prim_to_string kind)
        (args |> List.map expr_fingerprint |> String.concat ",")

let rec subst_var name replacement = function
  | LoomLambda.Var (var_name, _) as expr ->
      if String.equal var_name name then replacement else expr
  | FloatConst _ as expr -> expr
  | BoolConst _ as expr -> expr
  | Let (let_name, value, body) ->
      let value = subst_var name replacement value in
      if String.equal let_name name then Let (let_name, value, body)
      else Let (let_name, value, subst_var name replacement body)
  | If (cond, then_expr, else_expr) ->
      If
        ( subst_var name replacement cond,
          subst_var name replacement then_expr,
          subst_var name replacement else_expr )
  | Lambda (params, body) ->
      if
        List.exists
          (fun (param : LoomLambda.param) -> String.equal param.name name)
          params
      then Lambda (params, body)
      else Lambda (params, subst_var name replacement body)
  | Apply (fn, args) ->
      Apply
        (subst_var name replacement fn, List.map (subst_var name replacement) args)
  | Prim (prim, args) ->
      Prim (prim, List.map (subst_var name replacement) args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            let body =
              if
                List.exists
                  (fun (param : LoomLambda.param) -> String.equal param.name name)
                  params
              then body
              else subst_var name replacement body
            in
            Lambda (params, body) :: List.map (subst_var name replacement) tl
        | _ -> List.map (subst_var name replacement) args
      in
      TensorPrim (kind, args)

let rec arith_reassociate = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) ->
      Let (name, arith_reassociate value, arith_reassociate body)
  | If (cond, then_expr, else_expr) ->
      If
        ( arith_reassociate cond,
          arith_reassociate then_expr,
          arith_reassociate else_expr )
  | Lambda (params, body) -> Lambda (params, arith_reassociate body)
  | Apply (fn, args) ->
      Apply (arith_reassociate fn, List.map arith_reassociate args)
  | Prim (((FAdd | FMul) as prim), args) ->
      let rec flatten acc = function
        | LoomLambda.Prim (nested, nested_args) when nested = prim ->
            List.fold_left flatten acc nested_args
        | expr -> acc @ [ arith_reassociate expr ]
      in
      let args = List.fold_left flatten [] args in
      let args =
        List.sort
          (fun a b -> String.compare (expr_fingerprint a) (expr_fingerprint b))
          args
      in
      begin
        match args with
        | [] -> Prim (prim, [])
        | [ single ] -> single
        | lhs :: rhs :: tl ->
            List.fold_left (fun acc expr -> Prim (prim, [ acc; expr ]))
              (Prim (prim, [ lhs; rhs ])) tl
      end
  | Prim (((FMin | FMax | FCmpEq) as prim), [ lhs; rhs ]) ->
      let lhs = arith_reassociate lhs in
      let rhs = arith_reassociate rhs in
      if String.compare (expr_fingerprint lhs) (expr_fingerprint rhs) <= 0 then
        Prim (prim, [ lhs; rhs ])
      else Prim (prim, [ rhs; lhs ])
  | Prim (prim, args) -> Prim (prim, List.map arith_reassociate args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, arith_reassociate body)
            :: List.map arith_reassociate tl
        | _ -> List.map arith_reassociate args
      in
      TensorPrim (kind, args)

let rec scalar_cse seen = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) ->
      let value = scalar_cse seen value in
      if is_scalar_only_expr value then
        let key = expr_fingerprint value in
        begin
          match String_map.find_opt key seen with
          | Some replacement -> scalar_cse seen (subst_var name replacement body)
          | None ->
              let seen = String_map.add key value seen in
              Let (name, value, scalar_cse seen body)
        end
      else Let (name, value, scalar_cse seen body)
  | If (cond, then_expr, else_expr) ->
      If (scalar_cse seen cond, scalar_cse seen then_expr, scalar_cse seen else_expr)
  | Lambda (params, body) -> Lambda (params, scalar_cse seen body)
  | Apply (fn, args) -> Apply (scalar_cse seen fn, List.map (scalar_cse seen) args)
  | Prim (prim, args) -> Prim (prim, List.map (scalar_cse seen) args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, scalar_cse seen body) :: List.map (scalar_cse seen) tl
        | _ -> List.map (scalar_cse seen) args
      in
      TensorPrim (kind, args)

let branch_hoist = let_float

let rec dce = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) ->
      let value = dce value in
      let body = dce body in
      if String_set.mem name (free_vars body) then Let (name, value, body)
      else body
  | If (cond, then_expr, else_expr) ->
      If (dce cond, dce then_expr, dce else_expr)
  | Lambda (params, body) -> Lambda (params, dce body)
  | Apply (fn, args) -> Apply (dce fn, List.map dce args)
  | Prim (prim, args) -> Prim (prim, List.map dce args)
  | TensorPrim (kind, args) ->
      let args =
        match args with
        | Lambda (params, body) :: tl ->
            Lambda (params, dce body) :: List.map dce tl
        | _ -> List.map dce args
      in
      TensorPrim (kind, args)

let rec hoist_invariant_lets bound_vars = function
  | LoomLambda.Let (name, value, body) ->
      let value_hoisted, value = hoist_invariant_lets bound_vars value in
      let body_hoisted, body =
        hoist_invariant_lets (String_set.add name bound_vars) body
      in
      let current_let = LoomLambda.Let (name, value, body) in
      if
        is_scalar_only_expr value
        && String_set.disjoint bound_vars (free_vars value)
      then (value_hoisted @ [ (name, value) ] @ body_hoisted, body)
      else (value_hoisted @ body_hoisted, current_let)
  | LoomLambda.If (cond, then_expr, else_expr) ->
      let cond_hoisted, cond = hoist_invariant_lets bound_vars cond in
      let then_hoisted, then_expr = hoist_invariant_lets bound_vars then_expr in
      let else_hoisted, else_expr = hoist_invariant_lets bound_vars else_expr in
      let then_expr =
        List.fold_right
          (fun (name, value) acc -> LoomLambda.Let (name, value, acc))
          then_hoisted then_expr
      in
      let else_expr =
        List.fold_right
          (fun (name, value) acc -> LoomLambda.Let (name, value, acc))
          else_hoisted else_expr
      in
      (cond_hoisted, If (cond, then_expr, else_expr))
  | LoomLambda.Prim (prim, args) ->
      let hoisted, args =
        List.fold_left
          (fun (acc_hoisted, acc_args) arg ->
            let hoisted, arg = hoist_invariant_lets bound_vars arg in
            (acc_hoisted @ hoisted, acc_args @ [ arg ]))
          ([], []) args
      in
      (hoisted, Prim (prim, args))
  | LoomLambda.TensorPrim (kind, args) ->
      let hoisted, args =
        List.fold_left
          (fun (acc_hoisted, acc_args) arg ->
            let hoisted, arg = hoist_invariant_lets bound_vars arg in
            (acc_hoisted @ hoisted, acc_args @ [ arg ]))
          ([], []) args
      in
      (hoisted, TensorPrim (kind, args))
  | LoomLambda.Apply (fn, args) ->
      let fn_hoisted, fn = hoist_invariant_lets bound_vars fn in
      let hoisted, args =
        List.fold_left
          (fun (acc_hoisted, acc_args) arg ->
            let hoisted, arg = hoist_invariant_lets bound_vars arg in
            (acc_hoisted @ hoisted, acc_args @ [ arg ]))
          (fn_hoisted, []) args
      in
      (hoisted, Apply (fn, args))
  | LoomLambda.Lambda (params, body) ->
      let bound_vars =
        List.fold_left
          (fun acc (param : LoomLambda.param) -> String_set.add param.name acc)
          bound_vars params
      in
      let hoisted, body = hoist_invariant_lets bound_vars body in
      let body =
        List.fold_right
          (fun (name, value) acc -> LoomLambda.Let (name, value, acc))
          hoisted body
      in
      ([], Lambda (params, body))
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> ([], expr)

let rec scalar_hoist = function
  | (LoomLambda.Var _ | FloatConst _ | BoolConst _) as expr -> expr
  | Let (name, value, body) -> Let (name, scalar_hoist value, scalar_hoist body)
  | If (cond, then_expr, else_expr) ->
      If (scalar_hoist cond, scalar_hoist then_expr, scalar_hoist else_expr)
  | Lambda (params, body) -> Lambda (params, scalar_hoist body)
  | Apply (fn, args) -> Apply (scalar_hoist fn, List.map scalar_hoist args)
  | Prim (prim, args) -> Prim (prim, List.map scalar_hoist args)
  | TensorPrim
      ( ((LoomLambda.TensorMap | TensorMap2) as kind),
        Lambda (params, body) :: tl ) ->
      let bound_vars =
        List.fold_left
          (fun acc (param : LoomLambda.param) -> String_set.add param.name acc)
          String_set.empty params
      in
      let hoisted, body = hoist_invariant_lets bound_vars (scalar_hoist body) in
      let expr =
        TensorPrim (kind, Lambda (params, body) :: List.map scalar_hoist tl)
      in
      List.fold_right
        (fun (name, value) acc ->
          LoomLambda.Let (name, scalar_hoist value, acc))
        hoisted expr
  | TensorPrim (kind, args) -> TensorPrim (kind, List.map scalar_hoist args)
