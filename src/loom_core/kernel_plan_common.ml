type launch_bucket = { max_n : int; block_size : int; num_warps : int }

type complexity_bucket =
  | Tiny
  | Small
  | Medium
  | Large

type producer_strategy =
  | FusedProducer
  | ClonedProducer
  | MaterializedProducer

type reduction_strategy =
  | FixedStrategy
  | DirectReduction
  | SmallDirectReduction
  | SmallPartialReduction
  | SingleStagePartialReduction
  | MultiStageTreeReduction
  | TwoPhaseThresholdedReduction

type storage_class =
  | OutputStorage
  | TemporaryStorage of int

type pointwise_class =
  | PointwiseFastPath
  | GeneralPointwise

type body_traits = string list

type elementwise_step = {
  node_id : int;
  kernel_name : string;
  output : string;
  inputs : (string * string) list;
  scalar_params : string list;
  block_size : int;
  num_warps : int;
  launch_buckets : launch_bucket list;
  plan_class : string;
  pointwise_class : pointwise_class;
  traits : body_traits;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type reduction_source =
  | PlainInput of string
  | MappedInput of {
      inputs : (string * string) list;
      scalar_params : string list;
      body : Tensor_ir.scalar_expr;
    }

type reduction_step = {
  node_id : int;
  kernel_name : string;
  combine_kernel_name : string;
  output : string;
  reduce_kind : Tensor_ir.reduce_kind;
  block_size : int;
  num_warps : int;
  launch_buckets : launch_bucket list;
  single_block_threshold : int option;
  small_reduction_threshold : int option;
  small_program_count : int option;
  source : reduction_source;
  reduction_family : string;
  reduction_class : string;
  traits : body_traits;
  strategy_kind : string;
  reduction_strategy : reduction_strategy;
  stage_count : int;
  stage_layout : string;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type step = Elementwise of elementwise_step | Reduction of reduction_step

type t = {
  entry_name : string;
  steps : step list;
  result_name : string;
  temporary_count : int;
}

module Int_map = Map.Make (Int)
module Int_set = Set.Make (Int)

let default_block_size = 256
let default_num_warps = 4

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

let complexity_bucket_of_score score =
  if score <= 2 then Tiny
  else if score <= 6 then Small
  else if score <= 14 then Medium
  else Large

let complexity_bucket_to_string = function
  | Tiny -> "tiny"
  | Small -> "small"
  | Medium -> "medium"
  | Large -> "large"

let reduction_body_score (metrics : Tensor_ir.body_metrics) =
  metrics.scalar_complexity + metrics.branch_count + metrics.div_count
  + metrics.repeated_subexpressions

let producer_strategy_to_string = function
  | FusedProducer -> "fused"
  | ClonedProducer -> "cloned"
  | MaterializedProducer -> "materialized"

let reduction_strategy_to_string = function
  | FixedStrategy -> "fixed"
  | DirectReduction -> "direct"
  | SmallDirectReduction -> "small-direct"
  | SmallPartialReduction -> "small-partial"
  | SingleStagePartialReduction -> "single-stage-partial"
  | MultiStageTreeReduction -> "multi-stage-tree"
  | TwoPhaseThresholdedReduction -> "two-phase-thresholded"

let pointwise_class_to_string = function
  | PointwiseFastPath -> "pointwise-fast-path"
  | GeneralPointwise -> "general-pointwise"

let traits_to_yojson traits =
  `List (List.map (fun name -> `String name) traits)

let add_trait enabled name traits = if enabled then name :: traits else traits

let unique_traits traits = traits |> List.sort_uniq String.compare

let has_trait name traits = List.exists (String.equal name) traits

let storage_class_to_yojson = function
  | OutputStorage -> `String "output"
  | TemporaryStorage slot ->
      `Assoc [ ("kind", `String "temporary"); ("slot", `Int slot) ]

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

let rec scalar_expr_has_mul = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, _, _) -> true
  | SUnary (_, expr) -> scalar_expr_has_mul expr
  | SBinary (_, lhs, rhs) -> scalar_expr_has_mul lhs || scalar_expr_has_mul rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_has_mul cond || scalar_expr_has_mul then_expr
      || scalar_expr_has_mul else_expr
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let rec scalar_expr_has_select = function
  | Tensor_ir.SSelect _ -> true
  | SUnary (_, expr) -> scalar_expr_has_select expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_has_select lhs || scalar_expr_has_select rhs
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let scalar_expr_has_minmax =
  let rec loop = function
    | Tensor_ir.SBinary ((Tensor_ir.Min | Tensor_ir.Max), _, _) -> true
    | SUnary (_, expr) -> loop expr
    | SBinary (_, lhs, rhs) -> loop lhs || loop rhs
    | SSelect (cond, then_expr, else_expr) ->
        loop cond || loop then_expr || loop else_expr
    | SVar _ | SConstF32 _ | SConstBool _ -> false
  in
  loop

let scalar_expr_has_addsub =
  let rec loop = function
    | Tensor_ir.SBinary ((Tensor_ir.Add | Tensor_ir.Sub), _, _) -> true
    | SUnary (_, expr) -> loop expr
    | SBinary (_, lhs, rhs) -> loop lhs || loop rhs
    | SSelect (cond, then_expr, else_expr) ->
        loop cond || loop then_expr || loop else_expr
    | SVar _ | SConstF32 _ | SConstBool _ -> false
  in
  loop

let pointwise_traits input_count scalar_params (metrics : Tensor_ir.body_metrics)
    body =
  []
  |> add_trait
       (input_count = 1 && scalar_params <> [] && metrics.has_branch)
       "threshold"
  |> add_trait
       (input_count = 1 && scalar_params = []
       && (metrics.has_branch || scalar_expr_has_minmax body))
       "simple-activation"
  |> add_trait (input_count = 2 && metrics.has_div && not metrics.has_branch)
       "ratio-book"
  |> add_trait (input_count = 2 && metrics.has_div && metrics.has_branch)
       "mixed-filter-ratio"
  |> add_trait (input_count = 1 && metrics.has_branch) "activation-or-threshold"
  |> add_trait (input_count = 2 && metrics.has_branch) "filter-or-book"
  |> add_trait (input_count = 2 && not metrics.has_branch) "affine-vector-update"
  |> add_trait (scalar_params <> [] && scalar_expr_has_addsub body) "affine"
  |> add_trait (scalar_expr_has_mul body) "mul"
  |> add_trait (scalar_expr_has_div body) "ratio"
  |> add_trait (scalar_expr_has_select body) "branch"
  |> add_trait (scalar_expr_has_minmax body) "clip"
  |> unique_traits

let is_delta_square_body = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs) ->
      lhs = rhs
      &&
      (match lhs with
      | Tensor_ir.SBinary (Tensor_ir.Sub, _, _) -> true
      | _ -> false)
  | _ -> false

let rec is_square_of_var name = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, Tensor_ir.SVar lhs, Tensor_ir.SVar rhs) ->
      String.equal lhs name && String.equal rhs name
  | _ -> false

let rec is_mul_of_vars lhs_name rhs_name = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, Tensor_ir.SVar lhs, Tensor_ir.SVar rhs) ->
      (String.equal lhs lhs_name && String.equal rhs rhs_name)
      || (String.equal lhs rhs_name && String.equal rhs lhs_name)
  | _ -> false

let rec scalar_expr_uses_name name = function
  | Tensor_ir.SVar value -> String.equal value name
  | SConstF32 _ | SConstBool _ -> false
  | SUnary (_, expr) -> scalar_expr_uses_name name expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_uses_name name lhs || scalar_expr_uses_name name rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_uses_name name cond || scalar_expr_uses_name name then_expr
      || scalar_expr_uses_name name else_expr

let is_norm_square_body = function
  | Tensor_ir.MappedInput { inputs = [ { name; _ } ]; scalar_params = []; body }
    ->
      is_square_of_var name body
  | _ -> false

let is_affine_norm_square_body = function
  | Tensor_ir.MappedInput { inputs = [ { name; _ } ]; scalar_params; body } ->
      List.exists (String.equal "scale") scalar_params
      && List.exists (String.equal "bias") scalar_params
      &&
      let is_scaled_input = function
        | Tensor_ir.SBinary (Tensor_ir.Mul, Tensor_ir.SVar lhs, Tensor_ir.SVar rhs)
          ->
            (String.equal lhs "scale" && String.equal rhs name)
            || (String.equal lhs name && String.equal rhs "scale")
        | _ -> false
      in
      let is_bias = function
        | Tensor_ir.SVar value -> String.equal value "bias"
        | _ -> false
      in
      let is_affine = function
        | Tensor_ir.SBinary (Tensor_ir.Add, lhs, rhs) ->
            (is_scaled_input lhs && is_bias rhs)
            || (is_bias lhs && is_scaled_input rhs)
        | _ -> false
      in
      (match body with
      | Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs) when lhs = rhs ->
          is_affine lhs
      | _ -> false)
  | _ -> false

let is_dot_product_body = function
  | Tensor_ir.MappedInput
      { inputs = [ { name = lhs; _ }; { name = rhs; _ } ]; scalar_params = [];
        body } ->
      is_mul_of_vars lhs rhs body
  | _ -> false

let is_weighted_product_body = function
  | Tensor_ir.MappedInput
      { inputs = [ { name = lhs; _ }; { name = rhs; _ } ]; scalar_params; body }
    ->
      scalar_params <> [] && not (is_delta_square_body body)
      && scalar_expr_uses_name lhs body && scalar_expr_uses_name rhs body
  | _ -> false

let scalar_params_contain names required =
  List.exists (String.equal required) names

let rec scalar_expr_has_outer_cap cap = function
  | Tensor_ir.SBinary (Tensor_ir.Min, _, Tensor_ir.SVar cap_name)
    when String.equal cap_name cap ->
      true
  | Tensor_ir.SBinary (Tensor_ir.Min, Tensor_ir.SVar cap_name, _)
    when String.equal cap_name cap ->
      true
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary (Tensor_ir.CmpGt, _, Tensor_ir.SVar cap_name),
        Tensor_ir.SVar then_name,
        _ )
    when String.equal cap_name cap && String.equal then_name cap ->
      true
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_has_outer_cap cap cond || scalar_expr_has_outer_cap cap then_expr
      || scalar_expr_has_outer_cap cap else_expr
  | SUnary (_, expr) -> scalar_expr_has_outer_cap cap expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_has_outer_cap cap lhs || scalar_expr_has_outer_cap cap rhs
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let is_clipped_robust_body scalar_params body =
  scalar_params_contain scalar_params "delta"
  && scalar_params_contain scalar_params "cap"
  && scalar_expr_uses_name "delta" body
  && scalar_expr_has_outer_cap "cap" body

let is_robust_body scalar_params body =
  scalar_params_contain scalar_params "delta"
  && (not (scalar_params_contain scalar_params "cap"))
  && scalar_expr_uses_name "delta" body

let specialized_reduction_family ~enable_norm ~enable_dot ~enable_weighted
    reduction_family (metrics : Tensor_ir.body_metrics) source =
  match (reduction_family, source) with
  | ("mapped", source)
    when enable_norm && is_norm_square_body source ->
      "norm-square"
  | (("mapped" | "weighted"), source)
    when enable_norm && is_affine_norm_square_body source ->
      "affine-norm-square"
  | ("mapped", source) when enable_dot && is_dot_product_body source ->
      "dot-product"
  | ("weighted", source) when enable_weighted && is_weighted_product_body source ->
      "weighted-product"
  | ("branchy", Tensor_ir.MappedInput { scalar_params; body; _ })
    when is_clipped_robust_body scalar_params body ->
      "clipped-robust"
  | ("branchy", Tensor_ir.MappedInput { scalar_params; body; _ })
    when is_robust_body scalar_params body ->
      "robust"
  | (("mapped" | "weighted"), Tensor_ir.MappedInput { body; _ })
    when (not metrics.has_branch) && (not metrics.has_div)
         && is_delta_square_body body ->
      "delta-square"
  | _ -> reduction_family

let reduction_traits reduction_family source (metrics : Tensor_ir.body_metrics) =
  let body_and_inputs =
    match source with
    | Tensor_ir.PlainInput _ -> None
    | Tensor_ir.MappedInput { inputs; scalar_params; body } ->
        Some (inputs, scalar_params, body)
  in
  let body =
    match body_and_inputs with Some (_, _, body) -> Some body | None -> None
  in
  []
  |> add_trait
       (String.equal reduction_family "norm-square"
       || String.equal reduction_family "affine-norm-square")
       "square"
  |> add_trait (String.equal reduction_family "dot-product") "dot"
  |> add_trait (String.equal reduction_family "weighted-product") "weighted"
  |> add_trait (String.equal reduction_family "delta-square") "delta-square"
  |> add_trait (String.equal reduction_family "ratio") "ratio"
  |> add_trait (String.equal reduction_family "robust") "robust"
  |> add_trait (String.equal reduction_family "clipped-robust") "clipped"
  |> add_trait metrics.has_branch "branch"
  |> add_trait metrics.has_div "ratio"
  |> add_trait
       (match body with Some body -> scalar_expr_has_mul body | None -> false)
       "mul"
  |> add_trait
       (match body with Some body -> scalar_expr_has_select body | None -> false)
       "branch-body"
  |> add_trait
       (match body with Some body -> scalar_expr_has_minmax body | None -> false)
       "clip-body"
  |> add_trait
       (match body_and_inputs with
       | Some (inputs, scalar_params, body) ->
           List.length inputs >= 2 && scalar_params <> []
           && scalar_expr_has_addsub body
       | None -> false)
       "pipeline-expanded"
  |> unique_traits

let launch_bucket_of_config (bucket : Optimizations.launch_bucket) =
  {
    max_n = bucket.max_n;
    block_size = bucket.block_size;
    num_warps = bucket.num_warps;
  }

let launch_buckets_from_config buckets =
  List.map launch_bucket_of_config buckets

let reduction_refs = function
  | Tensor_ir.PlainInput input -> [ input ]
  | MappedInput { inputs; _ } ->
      List.map (fun (binding : Tensor_ir.input_binding) -> binding.source) inputs

let use_counts program =
  let add_ref counts = function
    | Tensor_ir.ParamRef _ -> counts
    | Tensor_ir.NodeRef id ->
        let count = Option.value (Int_map.find_opt id counts) ~default:0 in
        Int_map.add id (count + 1) counts
  in
  let counts =
    List.fold_left
      (fun counts -> function
        | Tensor_ir.Elementwise1D { inputs; _ } ->
            List.fold_left
              (fun acc (binding : Tensor_ir.input_binding) ->
                add_ref acc binding.source)
              counts inputs
        | Reduce1D { source; _ } ->
            List.fold_left add_ref counts (reduction_refs source))
      Int_map.empty program.Tensor_ir.nodes
  in
  match program.result with
  | Tensor_ir.TensorResult value | Tensor_ir.ScalarResult value ->
      add_ref counts value

let result_node_id = function
  | Tensor_ir.TensorResult (Tensor_ir.NodeRef id)
  | Tensor_ir.ScalarResult (Tensor_ir.NodeRef id) ->
      Some id
  | _ -> None

let source_name_of_ref _plan = function
  | Tensor_ir.ParamRef name -> name
  | Tensor_ir.NodeRef id -> Printf.sprintf "tmp%d" id

let find_node program node_id =
  match
    List.find_opt
      (function
        | Tensor_ir.Elementwise1D { id; _ } | Reduce1D { id; _ } -> id = node_id)
      program.Tensor_ir.nodes
  with
  | Some node -> node
  | None -> invalid_arg "missing tensor ir node for kernel plan"

let producer_strategy_of_hint = function
  | Tensor_ir.NoPreference -> MaterializedProducer
  | PreferFuse -> FusedProducer
  | PreferClone -> ClonedProducer
  | PreferMaterialize -> MaterializedProducer

let reduce_kind_to_string = function
  | Tensor_ir.Sum -> "sum"
  | Tensor_ir.MaxReduce -> "max"

let launch_bucket_to_yojson bucket =
  `Assoc
    [
      ("max_n", `Int bucket.max_n);
      ("block_size", `Int bucket.block_size);
      ("num_warps", `Int bucket.num_warps);
    ]

let complexity_bucket_to_yojson bucket =
  `String (complexity_bucket_to_string bucket)

let reduction_source_to_yojson = function
  | PlainInput input ->
      `Assoc [ ("kind", `String "plain"); ("input", `String input) ]
  | MappedInput { inputs; scalar_params; body } ->
      `Assoc
        [
          ("kind", `String "mapped");
          ( "inputs",
            `List
              (List.map
                 (fun (name, source) ->
                   `Assoc [ ("name", `String name); ("source", `String source) ])
                 inputs) );
          ( "scalar_params",
            `List (List.map (fun name -> `String name) scalar_params) );
          ("body", Tensor_ir.scalar_expr_to_yojson body);
        ]
