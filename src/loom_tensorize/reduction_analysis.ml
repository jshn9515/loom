open Tensor_ir

let rec scalar_expr_rank = function
  | Tensor_ir.SVar name -> "v:" ^ name
  | SConstF32 value -> Printf.sprintf "f:%0.12g" value
  | SConstBool value -> "b:" ^ string_of_bool value
  | SUnary (_, expr) -> "u:" ^ scalar_expr_rank expr
  | SBinary (op, lhs, rhs) ->
      let op =
        match op with
        | Tensor_ir.Add -> "add"
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
      in
      Printf.sprintf "b:%s:%s:%s" op (scalar_expr_rank lhs)
        (scalar_expr_rank rhs)
  | SSelect (cond, then_expr, else_expr) ->
      Printf.sprintf "s:%s:%s:%s" (scalar_expr_rank cond)
        (scalar_expr_rank then_expr)
        (scalar_expr_rank else_expr)

let rec canonicalize_scalar_expr = function
  | Tensor_ir.SUnary (op, expr) -> SUnary (op, canonicalize_scalar_expr expr)
  | SBinary (((Add | Mul | Min | Max | CmpEq) as op), lhs, rhs) ->
      let lhs = canonicalize_scalar_expr lhs in
      let rhs = canonicalize_scalar_expr rhs in
      if String.compare (scalar_expr_rank lhs) (scalar_expr_rank rhs) <= 0 then
        SBinary (op, lhs, rhs)
      else SBinary (op, rhs, lhs)
  | SBinary (op, lhs, rhs) ->
      SBinary (op, canonicalize_scalar_expr lhs, canonicalize_scalar_expr rhs)
  | SSelect (cond, then_expr, else_expr) ->
      let cond = canonicalize_scalar_expr cond in
      let then_expr = canonicalize_scalar_expr then_expr in
      let else_expr = canonicalize_scalar_expr else_expr in
      let minmax op lhs rhs = canonicalize_scalar_expr (SBinary (op, lhs, rhs)) in
      begin
        match cond with
        | SBinary (CmpGt, lhs, rhs) when then_expr = lhs && else_expr = rhs ->
            minmax Max lhs rhs
        | SBinary (CmpGt, lhs, rhs) when then_expr = rhs && else_expr = lhs ->
            minmax Min lhs rhs
        | SBinary (CmpLt, lhs, rhs) when then_expr = lhs && else_expr = rhs ->
            minmax Min lhs rhs
        | SBinary (CmpLt, lhs, rhs) when then_expr = rhs && else_expr = lhs ->
            minmax Max lhs rhs
        | _ -> SSelect (cond, then_expr, else_expr)
      end
  | (SVar _ | SConstF32 _ | SConstBool _) as expr -> expr

let rec flatten_mul_terms = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs) ->
      flatten_mul_terms lhs @ flatten_mul_terms rhs
  | expr -> [ expr ]

let rebuild_mul_terms = function
  | [] -> Tensor_ir.SConstF32 1.0
  | term :: rest ->
      List.fold_left (fun acc item -> Tensor_ir.SBinary (Tensor_ir.Mul, acc, item))
        term rest

let is_scalar_param scalar_params = function
  | Tensor_ir.SVar name -> List.exists (String.equal name) scalar_params
  | _ -> false

let normalize_mul_expr scalar_params expr =
  let scalar_terms, data_terms =
    flatten_mul_terms expr
    |> List.map canonicalize_scalar_expr
    |> List.partition (is_scalar_param scalar_params)
  in
  let order terms =
    List.sort
      (fun lhs rhs ->
        String.compare (scalar_expr_rank lhs) (scalar_expr_rank rhs))
      terms
  in
  rebuild_mul_terms (order data_terms @ order scalar_terms)

let rec canonicalize_reduction_body scalar_params = function
  | Tensor_ir.SUnary (op, expr) ->
      SUnary (op, canonicalize_reduction_body scalar_params expr)
  | SBinary (Tensor_ir.Mul, lhs, rhs) ->
      normalize_mul_expr scalar_params
        (SBinary
           ( Tensor_ir.Mul,
             canonicalize_reduction_body scalar_params lhs,
             canonicalize_reduction_body scalar_params rhs ))
  | SBinary (op, lhs, rhs) ->
      canonicalize_scalar_expr
        (SBinary
           ( op,
             canonicalize_reduction_body scalar_params lhs,
             canonicalize_reduction_body scalar_params rhs ))
  | SSelect (cond, then_expr, else_expr) ->
      canonicalize_scalar_expr
        (SSelect
           ( canonicalize_reduction_body scalar_params cond,
             canonicalize_reduction_body scalar_params then_expr,
             canonicalize_reduction_body scalar_params else_expr ))
  | (SVar _ | SConstF32 _ | SConstBool _) as expr -> expr

let branchy_weighted_rescue_body scalar_params body =
  let body = canonicalize_reduction_body scalar_params body in
  let split_branch_term expr =
    let factors = flatten_mul_terms expr |> List.map canonicalize_scalar_expr in
    let scalar_factors, data_factors =
      List.partition (is_scalar_param scalar_params) factors
    in
    match (data_factors, scalar_factors) with
    | [], _ | _, [] -> None
    | data_factors, [ scalar_weight ] ->
        Some (rebuild_mul_terms data_factors, scalar_weight)
    | _ -> None
  in
  match body with
  | Tensor_ir.SSelect (cond, then_expr, else_expr) -> (
      match (split_branch_term then_expr, split_branch_term else_expr) with
      | Some (shared_then, then_weight), Some (shared_else, else_weight)
        when shared_then = shared_else ->
          rebuild_mul_terms
            [
              shared_then;
              SSelect
                ( canonicalize_scalar_expr cond,
                  then_weight,
                  else_weight );
            ]
      | _ -> body)
  | _ -> body

let float_of_const = function
  | Tensor_ir.SConstF32 value -> Some value
  | _ -> None

let bool_of_const = function
  | Tensor_ir.SConstBool value -> Some value
  | _ -> None

let rec scalar_expr_complexity = function
  | Tensor_ir.SVar _ | SConstF32 _ | SConstBool _ -> 0
  | SUnary (_, expr) -> 1 + scalar_expr_complexity expr
  | SBinary (_, lhs, rhs) ->
      1 + scalar_expr_complexity lhs + scalar_expr_complexity rhs
  | SSelect (cond, then_expr, else_expr) ->
      1
      + scalar_expr_complexity cond
      + scalar_expr_complexity then_expr
      + scalar_expr_complexity else_expr

let rec scalar_expr_has_branch = function
  | Tensor_ir.SSelect _ -> true
  | SUnary (_, expr) -> scalar_expr_has_branch expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_has_branch lhs || scalar_expr_has_branch rhs
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let rec scalar_expr_has_div = function
  | Tensor_ir.SBinary (Tensor_ir.Div, _, _) -> true
  | SUnary (_, expr) -> scalar_expr_has_div expr
  | SBinary (_, lhs, rhs) -> scalar_expr_has_div lhs || scalar_expr_has_div rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_has_div cond || scalar_expr_has_div then_expr
      || scalar_expr_has_div else_expr
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let rec scalar_expr_branch_count = function
  | Tensor_ir.SSelect (cond, then_expr, else_expr) ->
      1 + scalar_expr_branch_count cond + scalar_expr_branch_count then_expr
      + scalar_expr_branch_count else_expr
  | SUnary (_, expr) -> scalar_expr_branch_count expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_branch_count lhs + scalar_expr_branch_count rhs
  | SVar _ | SConstF32 _ | SConstBool _ -> 0

let rec scalar_expr_div_count = function
  | Tensor_ir.SBinary (Tensor_ir.Div, lhs, rhs) ->
      1 + scalar_expr_div_count lhs + scalar_expr_div_count rhs
  | SUnary (_, expr) -> scalar_expr_div_count expr
  | SBinary (_, lhs, rhs) -> scalar_expr_div_count lhs + scalar_expr_div_count rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_div_count cond + scalar_expr_div_count then_expr
      + scalar_expr_div_count else_expr
  | SVar _ | SConstF32 _ | SConstBool _ -> 0

let scalar_expr_repeated_subexpressions expr =
  let counts = Hashtbl.create 32 in
  let rec loop = function
    | (SVar _ | SConstF32 _ | SConstBool _) as node -> ignore node
    | (SUnary _ | SBinary _ | SSelect _) as node ->
        let key = scalar_expr_rank node in
        let count = Option.value (Hashtbl.find_opt counts key) ~default:0 in
        Hashtbl.replace counts key (count + 1);
        begin
          match node with
          | SUnary (_, inner) -> loop inner
          | SBinary (_, lhs, rhs) ->
              loop lhs;
              loop rhs
          | SSelect (cond, then_expr, else_expr) ->
              loop cond;
              loop then_expr;
              loop else_expr
          | _ -> ()
        end
  in
  loop expr;
  Hashtbl.fold
    (fun _ count acc -> if count > 1 then acc + (count - 1) else acc)
    counts 0

let body_metrics use_count body =
  {
    Tensor_ir.scalar_complexity = scalar_expr_complexity body;
    has_branch = scalar_expr_has_branch body;
    has_div = scalar_expr_has_div body;
    branch_count = scalar_expr_branch_count body;
    div_count = scalar_expr_div_count body;
    repeated_subexpressions = scalar_expr_repeated_subexpressions body;
    estimated_uses = use_count;
  }

type reduction_body_pattern =
  | GenericReductionBody
  | DotLikeBody
  | WeightedDotLikeBody
  | SquaredDifferenceBody
  | WeightedSquaredDifferenceBody

let classify_reduction_body_pattern scalar_params body =
  let body = canonicalize_reduction_body scalar_params body in
  let factors = flatten_mul_terms body in
  let scalar_factors, data_factors =
    List.partition (is_scalar_param scalar_params) factors
  in
  let is_tensor_term = function
    | Tensor_ir.SVar name ->
        not (List.exists (String.equal name) scalar_params)
    | _ -> false
  in
  let is_difference = function
    | Tensor_ir.SBinary (Tensor_ir.Sub, lhs, rhs) -> lhs <> rhs
    | _ -> false
  in
  match (scalar_factors, data_factors) with
  | [], [ lhs; rhs ] when is_tensor_term lhs && is_tensor_term rhs -> DotLikeBody
  | _ :: _, [ lhs; rhs ] when is_tensor_term lhs && is_tensor_term rhs ->
      WeightedDotLikeBody
  | [], [ lhs; rhs ] when lhs = rhs && is_difference lhs -> SquaredDifferenceBody
  | _ :: _, [ lhs; rhs ] when lhs = rhs && is_difference lhs ->
      WeightedSquaredDifferenceBody
  | _ -> GenericReductionBody

let adjusted_reduction_body_metrics scalar_params use_count body =
  let body = canonicalize_reduction_body scalar_params body in
  let metrics = body_metrics use_count body in
  match classify_reduction_body_pattern scalar_params body with
  | DotLikeBody ->
      { metrics with scalar_complexity = min metrics.scalar_complexity 1; repeated_subexpressions = 0 }
  | WeightedDotLikeBody ->
      { metrics with scalar_complexity = min metrics.scalar_complexity 2; repeated_subexpressions = 0 }
  | SquaredDifferenceBody ->
      {
        metrics with
        scalar_complexity = max 2 (metrics.scalar_complexity - 1);
        repeated_subexpressions = 0;
      }
  | WeightedSquaredDifferenceBody ->
      {
        metrics with
        scalar_complexity = max 3 (metrics.scalar_complexity - 1);
        repeated_subexpressions = 0;
      }
  | GenericReductionBody -> metrics

let classify_reduction_shape source =
  match source with
  | Tensor_ir.PlainInput _ -> Tensor_ir.PlainReduction
  | MappedInput { body; scalar_params; _ } ->
      if scalar_expr_has_branch body then Tensor_ir.BranchyMappedReduction
      else if scalar_expr_has_div body then Tensor_ir.RatioMappedReduction
      else if scalar_params <> [] then Tensor_ir.WeightedMappedReduction
      else Tensor_ir.MappedReduction

let rec simplify_scalar_expr = function
  | Tensor_ir.SUnary (op, expr) -> (
      let expr = simplify_scalar_expr expr in
      match (op, float_of_const expr) with
      | Tensor_ir.Neg, Some value -> SConstF32 (-.value)
      | Sqrt, Some value -> SConstF32 (sqrt value)
      | Exp, Some value -> SConstF32 (exp value)
      | Log, Some value -> SConstF32 (log value)
      | _ -> SUnary (op, expr))
  | SBinary (op, lhs, rhs) -> (
      let lhs = simplify_scalar_expr lhs in
      let rhs = simplify_scalar_expr rhs in
      match
        ( op,
          float_of_const lhs,
          float_of_const rhs,
          bool_of_const lhs,
          bool_of_const rhs )
      with
      | Add, Some lhs, Some rhs, _, _ -> SConstF32 (lhs +. rhs)
      | Sub, Some lhs, Some rhs, _, _ -> SConstF32 (lhs -. rhs)
      | Mul, Some lhs, Some rhs, _, _ -> SConstF32 (lhs *. rhs)
      | Div, Some lhs, Some rhs, _, _ -> SConstF32 (lhs /. rhs)
      | Min, Some lhs, Some rhs, _, _ -> SConstF32 (min lhs rhs)
      | Max, Some lhs, Some rhs, _, _ -> SConstF32 (max lhs rhs)
      | CmpLt, Some lhs, Some rhs, _, _ -> SConstBool (lhs < rhs)
      | CmpLe, Some lhs, Some rhs, _, _ -> SConstBool (lhs <= rhs)
      | CmpGt, Some lhs, Some rhs, _, _ -> SConstBool (lhs > rhs)
      | CmpGe, Some lhs, Some rhs, _, _ -> SConstBool (lhs >= rhs)
      | CmpEq, Some lhs, Some rhs, _, _ -> SConstBool (lhs = rhs)
      | CmpEq, _, _, Some lhs, Some rhs -> SConstBool (lhs = rhs)
      | Add, Some 0.0, _, _, _ -> rhs
      | Add, _, Some 0.0, _, _ -> lhs
      | Sub, _, Some 0.0, _, _ -> lhs
      | Mul, Some 0.0, _, _, _ | Mul, _, Some 0.0, _, _ -> SConstF32 0.0
      | Mul, Some 1.0, _, _, _ -> rhs
      | Mul, _, Some 1.0, _, _ -> lhs
      | Div, _, Some 1.0, _, _ -> lhs
      | Min, _, _, _, _ when lhs = rhs -> lhs
      | Max, _, _, _, _ when lhs = rhs -> lhs
      | _ -> SBinary (op, lhs, rhs))
  | SSelect (cond, then_expr, else_expr) -> (
      let cond = simplify_scalar_expr cond in
      let then_expr = simplify_scalar_expr then_expr in
      let else_expr = simplify_scalar_expr else_expr in
      match bool_of_const cond with
      | Some true -> then_expr
      | Some false -> else_expr
      | None when then_expr = else_expr -> then_expr
      | None -> SSelect (cond, then_expr, else_expr))
  | (SVar _ | SConstF32 _ | SConstBool _) as expr -> expr

let rec reduction_body_cse_expr expr =
  let expr =
    match expr with
    | Tensor_ir.SUnary (op, inner) -> SUnary (op, reduction_body_cse_expr inner)
    | SBinary (op, lhs, rhs) ->
        SBinary (op, reduction_body_cse_expr lhs, reduction_body_cse_expr rhs)
    | SSelect (cond, then_expr, else_expr) ->
        SSelect
          ( reduction_body_cse_expr cond,
            reduction_body_cse_expr then_expr,
            reduction_body_cse_expr else_expr )
    | (SVar _ | SConstF32 _ | SConstBool _) as leaf -> leaf
  in
  match expr with
  | SBinary (Add, lhs, rhs) when lhs = rhs ->
      simplify_scalar_expr (SBinary (Mul, SConstF32 2.0, lhs))
  | SBinary (Sub, lhs, rhs) when lhs = rhs -> SConstF32 0.0
  | SBinary (Mul, lhs, rhs) when lhs = rhs ->
      simplify_scalar_expr (SBinary (Mul, lhs, rhs))
  | SSelect (_, then_expr, else_expr) when then_expr = else_expr -> then_expr
  | _ -> expr

let reduction_sources = function
  | Tensor_ir.PlainInput input -> [ input ]
  | MappedInput { inputs; _ } ->
      List.map (fun (binding : Tensor_ir.input_binding) -> binding.source) inputs

let reduction_source_complexity = function
  | Tensor_ir.PlainInput _ -> 0
  | MappedInput { body; _ } -> scalar_expr_complexity body

let rec substitute_scalar_expr name replacement = function
  | Tensor_ir.SVar value when String.equal value name -> replacement
  | Tensor_ir.SUnary (op, expr) ->
      SUnary (op, substitute_scalar_expr name replacement expr)
  | SBinary (op, lhs, rhs) ->
      SBinary
        ( op,
          substitute_scalar_expr name replacement lhs,
          substitute_scalar_expr name replacement rhs )
  | SSelect (cond, then_expr, else_expr) ->
      SSelect
        ( substitute_scalar_expr name replacement cond,
          substitute_scalar_expr name replacement then_expr,
          substitute_scalar_expr name replacement else_expr )
  | (SVar _ | SConstF32 _ | SConstBool _) as expr -> expr

let rename_scalar_expr_vars renamings expr =
  List.fold_left
    (fun acc (from_name, to_name) ->
      substitute_scalar_expr from_name (Tensor_ir.SVar to_name) acc)
    expr renamings
