open LoomLambda
open Tensor_ir

module String_map = Map.Make (String)

type tensor_value = { source : Tensor_ir.value_ref; shape_symbol : string }

type compiled_value =
  | ScalarExpr of Tensor_ir.scalar_expr
  | ScalarRef of Tensor_ir.value_ref
  | Tensor of tensor_value

type state = { next_id : int; nodes : Tensor_ir.node list }

let empty_state = { next_id = 0; nodes = [] }

let empty_metrics =
  {
    Tensor_ir.scalar_complexity = 0;
    has_branch = false;
    has_div = false;
    branch_count = 0;
    div_count = 0;
    repeated_subexpressions = 0;
    estimated_uses = 0;
  }

let default_handling_hint = Materialization_policy.default_handling_hint

let add_node state mk_node =
  let id = state.next_id in
  let node = mk_node id in
  ({ next_id = id + 1; nodes = state.nodes @ [ node ] }, id)

let rec scalar_expr_of env loc = function
  | Var (name, _) -> (
      match String_map.find_opt name env with
      | Some (ScalarExpr expr) -> expr
      | Some (ScalarRef _) ->
          Diagnostic.raise_error ~loc
            "reduction results cannot be captured inside scalar lambdas"
      | Some (Tensor _) ->
          Diagnostic.raise_error ~loc
            "tensor capture inside scalar lambdas is not supported"
      | None ->
          Diagnostic.raise_error ~loc
            (Printf.sprintf "unbound scalar variable %s" name))
  | FloatConst value -> SConstF32 value
  | BoolConst value -> SConstBool value
  | Let (name, value, body) ->
      let value = scalar_expr_of env loc value in
      scalar_expr_of (String_map.add name (ScalarExpr value) env) loc body
  | If (cond, then_expr, else_expr) ->
      SSelect
        ( scalar_expr_of env loc cond,
          scalar_expr_of env loc then_expr,
          scalar_expr_of env loc else_expr )
  | Prim (prim, args) ->
      let args = List.map (scalar_expr_of env loc) args in
      let unary op =
        match args with
        | [ arg ] -> SUnary (op, arg)
        | _ -> Diagnostic.raise_error ~loc "invalid unary primitive arity"
      in
      let binary op =
        match args with
        | [ lhs; rhs ] -> SBinary (op, lhs, rhs)
        | _ -> Diagnostic.raise_error ~loc "invalid binary primitive arity"
      in
      begin
        match prim with
        | FAdd -> binary Add
        | FSub -> binary Sub
        | FMul -> binary Mul
        | FDiv -> binary Div
        | FNeg -> unary Neg
        | FMin -> binary Min
        | FMax -> binary Max
        | FSqrt -> unary Sqrt
        | FExp -> unary Exp
        | FLog -> unary Log
        | FCmpLt -> binary CmpLt
        | FCmpLe -> binary CmpLe
        | FCmpGt -> binary CmpGt
        | FCmpGe -> binary CmpGe
        | FCmpEq -> binary CmpEq
        | Select -> (
            match args with
            | [ cond; then_expr; else_expr ] ->
                SSelect (cond, then_expr, else_expr)
            | _ -> Diagnostic.raise_error ~loc "invalid select primitive arity")
      end
  | Apply (Var (name, _), _) ->
      Diagnostic.raise_error ~loc
        (Printf.sprintf "unknown function call inside scalar lambda: %s" name)
  | Apply _ ->
      Diagnostic.raise_error ~loc
        "unsupported higher-order function call inside scalar lambda"
  | TensorPrim _ ->
      Diagnostic.raise_error ~loc
        "tensor primitives are not allowed inside scalar lambdas"
  | Lambda _ ->
      Diagnostic.raise_error ~loc
        "nested lambdas are not supported inside scalar lambdas"

let shape_symbol = "n"

let rec compile_expr state env loc = function
  | Var (name, _) -> (
      match String_map.find_opt name env with
      | Some value -> (state, value)
      | None ->
          Diagnostic.raise_error ~loc
            (Printf.sprintf "unbound staged variable %s" name))
  | FloatConst value -> (state, ScalarExpr (SConstF32 value))
  | BoolConst value -> (state, ScalarExpr (SConstBool value))
  | Let (name, value, body) ->
      let state, value = compile_expr state env loc value in
      compile_expr state (String_map.add name value env) loc body
  | If (cond, then_expr, else_expr) ->
      let cond = scalar_expr_of env loc cond in
      let then_expr = scalar_expr_of env loc then_expr in
      let else_expr = scalar_expr_of env loc else_expr in
      (state, ScalarExpr (SSelect (cond, then_expr, else_expr)))
  | Prim _ as expr -> (state, ScalarExpr (scalar_expr_of env loc expr))
  | Apply _ ->
      Diagnostic.raise_error ~loc
        "unsupported top-level function application in staged entry"
  | Lambda _ ->
      Diagnostic.raise_error ~loc
        "unexpected lambda at top level during tensorization"
  | TensorPrim (TensorReduceSum, [ tensor_expr ]) ->
      let state, tensor = compile_tensor state env loc tensor_expr in
      let state, id =
        add_node state (fun id ->
            Reduce1D
              {
                id;
                shape_symbol = tensor.shape_symbol;
                source = PlainInput tensor.source;
                kind = Sum;
                metrics = empty_metrics;
                reduction_shape = PlainReduction;
                handling_hint = default_handling_hint;
              })
      in
      (state, ScalarRef (NodeRef id))
  | TensorPrim (TensorReduceMax, [ tensor_expr ]) ->
      let state, tensor = compile_tensor state env loc tensor_expr in
      let state, id =
        add_node state (fun id ->
            Reduce1D
              {
                id;
                shape_symbol = tensor.shape_symbol;
                source = PlainInput tensor.source;
                kind = MaxReduce;
                metrics = empty_metrics;
                reduction_shape = PlainReduction;
                handling_hint = default_handling_hint;
              })
      in
      (state, ScalarRef (NodeRef id))
  | TensorPrim (TensorMap, [ Lambda ([ param ], body); tensor_expr ]) ->
      let state, tensor = compile_tensor state env loc tensor_expr in
      let lambda_env =
        String_map.add param.name (ScalarExpr (SVar param.name)) env
      in
      let body = scalar_expr_of lambda_env loc body in
      let scalar_params =
        Tensor_ir.scalar_expr_free_vars body
        |> List.filter (fun name -> not (String.equal name param.name))
        |> List.sort_uniq String.compare
      in
      let state, id =
        add_node state (fun id ->
            Elementwise1D
              {
                id;
                shape_symbol = tensor.shape_symbol;
                inputs = [ { name = param.name; source = tensor.source } ];
                scalar_params;
                body;
                metrics = empty_metrics;
                handling_hint = default_handling_hint;
              })
      in
      (state, Tensor { source = NodeRef id; shape_symbol = tensor.shape_symbol })
  | TensorPrim (TensorMap2, [ Lambda ([ lhs; rhs ], body); a; b ]) ->
      let state, a = compile_tensor state env loc a in
      let state, b = compile_tensor state env loc b in
      if not (String.equal a.shape_symbol b.shape_symbol) then
        Diagnostic.raise_error ~loc
          "map2 requires tensors with the same symbolic extent";
      let lambda_env =
        env
        |> String_map.add lhs.name (ScalarExpr (SVar lhs.name))
        |> String_map.add rhs.name (ScalarExpr (SVar rhs.name))
      in
      let body = scalar_expr_of lambda_env loc body in
      let scalar_params =
        Tensor_ir.scalar_expr_free_vars body
        |> List.filter (fun name ->
               not (String.equal name lhs.name || String.equal name rhs.name))
        |> List.sort_uniq String.compare
      in
      let state, id =
        add_node state (fun id ->
            Elementwise1D
              {
                id;
                shape_symbol = a.shape_symbol;
                inputs =
                  [
                    { name = lhs.name; source = a.source };
                    { name = rhs.name; source = b.source };
                  ];
                scalar_params;
                body;
                metrics = empty_metrics;
                handling_hint = default_handling_hint;
              })
      in
      (state, Tensor { source = NodeRef id; shape_symbol = a.shape_symbol })
  | TensorPrim _ ->
      Diagnostic.raise_error ~loc "invalid tensor primitive application shape"

and compile_tensor state env loc expr =
  let state, value = compile_expr state env loc expr in
  match value with
  | Tensor tensor -> (state, tensor)
  | ScalarExpr _ | ScalarRef _ ->
      Diagnostic.raise_error ~loc "expected a tensor expression"

let env_of_params params =
  List.fold_left
    (fun env param ->
      match param.LoomLambda.ty with
      | Loom_types.LLFloat ->
          String_map.add param.name (ScalarExpr (SVar param.name)) env
      | Loom_types.LLTensor1F32 ->
          String_map.add param.name
            (Tensor { source = ParamRef param.name; shape_symbol })
            env
      | ty ->
          Diagnostic.raise_error
            (Printf.sprintf "unsupported entry parameter type %s"
               (Loom_types.stage_type_to_string ty)))
    String_map.empty params

let program_of_entry ?(optimizations = Optimizations.none)
    (entry : LoomLambda.entry) =
  let env = env_of_params entry.params in
  let state, result = compile_expr empty_state env Location.none entry.body in
  let params =
    List.map
      (fun param ->
        match param.LoomLambda.ty with
        | Loom_types.LLFloat -> ScalarF32 param.name
        | Loom_types.LLTensor1F32 -> Tensor1F32 (param.name, shape_symbol)
        | ty ->
            Diagnostic.raise_error
              (Printf.sprintf "unsupported entry parameter type %s"
                 (Loom_types.stage_type_to_string ty)))
      entry.params
  in
  let result =
    match result with
    | Tensor tensor -> TensorResult tensor.source
    | ScalarRef value -> ScalarResult value
    | ScalarExpr _ ->
        Diagnostic.raise_error
          "entry lowered to a scalar expression instead of a reduction result"
  in
  let program =
    { entry_name = entry.name; params; result; nodes = state.nodes }
  in
  let program =
    Tensor_passes.apply_optimizations ~optimizations ~empty_metrics program
  in
  Validate.program program;
  program
