type launch_bucket = Kernel_plan_common.launch_bucket = {
  max_n : int;
  block_size : int;
  num_warps : int;
}

type complexity_bucket = Kernel_plan_common.complexity_bucket =
  | Tiny
  | Small
  | Medium
  | Large

type producer_strategy = Kernel_plan_common.producer_strategy =
  | FusedProducer
  | ClonedProducer
  | MaterializedProducer

type reduction_strategy = Kernel_plan_common.reduction_strategy =
  | FixedStrategy
  | DirectReduction
  | SmallDirectReduction
  | SmallPartialReduction
  | SingleStagePartialReduction
  | MultiStageTreeReduction
  | TwoPhaseThresholdedReduction

type storage_class = Kernel_plan_common.storage_class =
  | OutputStorage
  | TemporaryStorage of int

type pointwise_class = Kernel_plan_common.pointwise_class =
  | PointwiseFastPath
  | GeneralPointwise

type reduction_source = Kernel_plan_common.reduction_source =
  | PlainInput of string
  | MappedInput of {
      inputs : (string * string) list;
      scalar_params : string list;
      body : Tensor_ir.scalar_expr;
    }

type cuda_pointwise_family =
  | GenericPointwiseKernel
  | SmallNPointwiseKernel
  | VectorizedPointwiseKernel

type cuda_reduction_execution =
  | WorkspaceTreeReduction
  | SingleBlockReduction
  | AtomicOutputReduction

type cuda_combine_family =
  | NoCombineKernel
  | SharedTreeCombine

type cuda_body_traits = string list

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
  pointwise_family : cuda_pointwise_family;
  traits : cuda_body_traits;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
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
  strategy_kind : string;
  reduction_strategy : reduction_strategy;
  execution_family : cuda_reduction_execution;
  combine_family : cuda_combine_family;
  traits : cuda_body_traits;
  uses_workspace : bool;
  stage_count : int;
  stage_layout : string;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type step =
  | Elementwise of elementwise_step
  | Reduction of reduction_step

type t = {
  entry_name : string;
  steps : step list;
  result_name : string;
  temporary_count : int;
}

let complexity_bucket_to_yojson = Kernel_plan_common.complexity_bucket_to_yojson
let producer_strategy_to_string = Kernel_plan_common.producer_strategy_to_string
let pointwise_class_to_string = Kernel_plan_common.pointwise_class_to_string
let reduction_strategy_to_string = Kernel_plan_common.reduction_strategy_to_string

let pointwise_family_to_string = function
  | GenericPointwiseKernel -> "generic-pointwise"
  | SmallNPointwiseKernel -> "small-n-pointwise"
  | VectorizedPointwiseKernel -> "vectorized-pointwise"

let reduction_execution_to_string = function
  | WorkspaceTreeReduction -> "workspace-tree"
  | SingleBlockReduction -> "single-block"
  | AtomicOutputReduction -> "atomic-output"

let combine_family_to_string = function
  | NoCombineKernel -> "none"
  | SharedTreeCombine -> "shared-tree-combine"

let reduction_source_to_yojson = Kernel_plan_common.reduction_source_to_yojson
let reduce_kind_to_string = Kernel_plan_common.reduce_kind_to_string
let storage_class_to_yojson = Kernel_plan_common.storage_class_to_yojson
let launch_bucket_to_yojson = Kernel_plan_common.launch_bucket_to_yojson

let traits_to_yojson traits =
  `List (List.map (fun name -> `String name) traits)

let add_trait enabled name traits = if enabled then name :: traits else traits

let unique_traits traits =
  traits |> List.sort_uniq String.compare

let rec scalar_expr_has_mul = function
  | Tensor_ir.SBinary (Tensor_ir.Mul, _, _) -> true
  | SUnary (_, expr) -> scalar_expr_has_mul expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_has_mul lhs || scalar_expr_has_mul rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_has_mul cond || scalar_expr_has_mul then_expr
      || scalar_expr_has_mul else_expr
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let rec scalar_expr_has_div = function
  | Tensor_ir.SBinary (Tensor_ir.Div, _, _) -> true
  | SUnary (_, expr) -> scalar_expr_has_div expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_has_div lhs || scalar_expr_has_div rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_has_div cond || scalar_expr_has_div then_expr
      || scalar_expr_has_div else_expr
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
  Kernel_plan_common.pointwise_traits input_count scalar_params metrics body

let reduction_traits reduction_family source (metrics : Tensor_ir.body_metrics) =
  Kernel_plan_common.reduction_traits reduction_family source metrics

let has_trait name traits = List.exists (String.equal name) traits

let vectorizable_ratio_tail traits =
  has_trait "ratio-book" traits || has_trait "mixed-filter-ratio" traits

let tail_tuned_vector_candidate input_count (metrics : Tensor_ir.body_metrics)
    traits =
  input_count > 0 && input_count <= 3
  && (not metrics.has_div
     || (metrics.scalar_complexity <= 12 && vectorizable_ratio_tail traits))

let selected_profile_generic_pointwise input_count traits =
  (input_count = 2 && has_trait "affine-vector-update" traits
  && not (has_trait "ratio" traits))
  || (input_count = 1 && has_trait "affine" traits && has_trait "clip" traits)
  || (input_count = 2 && has_trait "filter-or-book" traits
     && not (has_trait "ratio" traits))

let selected_profile_prevector_generic_pointwise input_count traits =
  (input_count = 2 && has_trait "affine-vector-update" traits
  && has_trait "ratio-book" traits
  && not (has_trait "mul" traits))
  || (input_count = 2 && has_trait "filter-or-book" traits
     && not (has_trait "ratio" traits))
  || (input_count = 2 && has_trait "mixed-filter-ratio" traits)

let affine_vector_tail_candidate input_count traits =
  input_count = 2 && has_trait "affine-vector-update" traits
  && not (has_trait "ratio" traits)

let book_filter_prefers_scalar (metrics : Tensor_ir.body_metrics) traits =
  (has_trait "filter-or-book" traits && not (has_trait "ratio" traits)
  && metrics.scalar_complexity >= 4)
  || (has_trait "mixed-filter-ratio" traits && metrics.scalar_complexity >= 8)

let pointwise_family_for_step optimizations entry_name pointwise_class
    complexity_bucket input_count (metrics : Tensor_ir.body_metrics) traits =
  ignore entry_name;
  if
    Optimizations.enabled optimizations Optimizations.CudaBookFilterRegisterPlan
    && book_filter_prefers_scalar metrics traits
  then GenericPointwiseKernel
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseAffineVectorTail
    && affine_vector_tail_candidate input_count traits
  then VectorizedPointwiseKernel
  else if
    Optimizations.enabled optimizations Optimizations.CudaSelectedProfile
    && selected_profile_prevector_generic_pointwise input_count traits
  then GenericPointwiseKernel
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseVectorize
    && (if
          Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
        then tail_tuned_vector_candidate input_count metrics traits
        else
          input_count > 0 && input_count <= 3 && not metrics.has_div
          && not (has_trait "ratio" traits))
  then VectorizedPointwiseKernel
  else
  if
    Optimizations.enabled optimizations Optimizations.CudaSelectedProfile
    && Optimizations.enabled optimizations
         Optimizations.CudaPointwiseInstructionSelect
    && selected_profile_generic_pointwise input_count traits
  then GenericPointwiseKernel
  else
  if
    Optimizations.enabled optimizations
      Optimizations.CudaPointwiseInstructionSelect
  then
    match (complexity_bucket, input_count, traits) with
    | (Tiny | Small), 1, traits
      when has_trait "activation-or-threshold" traits ->
        SmallNPointwiseKernel
    | Tiny, 1, _ -> SmallNPointwiseKernel
    | _ -> GenericPointwiseKernel
  else
  match
    ( Optimizations.enabled optimizations Optimizations.CudaPointwiseSmallShapePlan,
      pointwise_class,
      complexity_bucket,
      input_count,
      metrics.has_branch,
      metrics.has_div )
  with
  | true, _, (Tiny | Small), 1, _, _ -> SmallNPointwiseKernel
  | true, _, Tiny, 2, false, false -> SmallNPointwiseKernel
  | false, PointwiseFastPath, Tiny, _, _, _ -> SmallNPointwiseKernel
  | _ -> GenericPointwiseKernel

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
  | Tensor_ir.MappedInput { inputs = [ { name; _ } ]; scalar_params = []; body } ->
      is_square_of_var name body
  | _ -> false

let is_affine_norm_square_body = function
  | Tensor_ir.MappedInput { inputs = [ { name; _ } ]; scalar_params; body } ->
      List.exists (String.equal "scale") scalar_params
      && List.exists (String.equal "bias") scalar_params
      &&
      let is_scaled_input = function
        | Tensor_ir.SBinary
            ( Tensor_ir.Mul,
              Tensor_ir.SVar lhs,
              Tensor_ir.SVar rhs ) ->
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
  | Tensor_ir.MappedInput { inputs = [ { name = lhs; _ }; { name = rhs; _ } ]; scalar_params = []; body } ->
      is_mul_of_vars lhs rhs body
  | _ -> false

let is_not_delta_square_like body = not (is_delta_square_body body)

let is_weighted_product_body = function
  | Tensor_ir.MappedInput { inputs = [ { name = lhs; _ }; { name = rhs; _ } ]; scalar_params; body } ->
      scalar_params <> [] && is_not_delta_square_like body
      && scalar_expr_uses_name lhs body && scalar_expr_uses_name rhs body
  | _ -> false

let scalar_params_contain names required =
  List.exists (String.equal required) names

let rec scalar_expr_uses_var name = function
  | Tensor_ir.SVar value -> String.equal value name
  | SConstF32 _ | SConstBool _ -> false
  | SUnary (_, expr) -> scalar_expr_uses_var name expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_uses_var name lhs || scalar_expr_uses_var name rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_uses_var name cond || scalar_expr_uses_var name then_expr
      || scalar_expr_uses_var name else_expr

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
  && scalar_expr_uses_var "delta" body
  && scalar_expr_has_outer_cap "cap" body

let is_robust_body scalar_params body =
  scalar_params_contain scalar_params "delta"
  && (not (scalar_params_contain scalar_params "cap"))
  && scalar_expr_uses_var "delta" body

let kind_of_source_is_sum_like = function
  | Tensor_ir.MappedInput _ -> true
  | Tensor_ir.PlainInput _ -> false

let backend_reduction_family optimizations reduction_family
    (metrics : Tensor_ir.body_metrics) source =
  Kernel_plan_common.specialized_reduction_family
    ~enable_norm:
      (Optimizations.enabled optimizations Optimizations.CudaReductionNormPlan)
    ~enable_dot:
      (Optimizations.enabled optimizations Optimizations.CudaReductionDotPlan)
    ~enable_weighted:
      (Optimizations.enabled optimizations
         Optimizations.CudaReductionWeightedProductPlan)
    reduction_family metrics source

let selected_reduction_family optimizations entry_name reduction_family =
  ignore optimizations;
  ignore entry_name;
  reduction_family

let reduction_execution_for_step optimizations reduction_family complexity_bucket
    (metrics : Tensor_ir.body_metrics) kind reduction_strategy =
  match (kind, reduction_family, complexity_bucket, reduction_strategy) with
  | Tensor_ir.Sum, "robust", Large, _
    when Optimizations.enabled optimizations Optimizations.CudaReductionTailTune
         && metrics.scalar_complexity <= 32 ->
      AtomicOutputReduction
  | ( Tensor_ir.Sum,
      ("robust" | "clipped-robust" | "branchy" | "ratio"),
      (Small | Medium),
      _ )
    when Optimizations.enabled optimizations Optimizations.CudaReductionTailTune
         && metrics.scalar_complexity <= 12 ->
      AtomicOutputReduction
  | ( Tensor_ir.Sum,
      ("robust" | "clipped-robust" | "branchy"),
      Medium,
      _ )
    when Optimizations.enabled optimizations
           Optimizations.CudaReductionMediumTailPlan
         && metrics.scalar_complexity <= 18 ->
      AtomicOutputReduction
  | Tensor_ir.Sum, "clipped-robust", Large, _
    when Optimizations.enabled optimizations
           Optimizations.CudaReductionMediumTailPlan
         && metrics.scalar_complexity <= 40 ->
      AtomicOutputReduction
  | Tensor_ir.Sum, "branchy", Large, _
    when Optimizations.enabled optimizations
           Optimizations.CudaReductionMediumTailPlan
         && metrics.has_div && metrics.scalar_complexity <= 14 ->
      AtomicOutputReduction
  | ( Tensor_ir.Sum,
      ( "norm-square" | "affine-norm-square" | "dot-product" | "delta-square"
      | "weighted-product" ),
      (Tiny | Small | Medium),
      _ )
    when (not metrics.has_branch) && (not metrics.has_div) ->
      AtomicOutputReduction
  | ( Tensor_ir.Sum,
      "mapped",
      (Tiny | Small | Medium),
      _ )
    when (not metrics.has_branch) && (not metrics.has_div)
         && metrics.scalar_complexity >= 2 ->
      AtomicOutputReduction
  | (_, _, _, (DirectReduction | SmallDirectReduction)) -> SingleBlockReduction
  | _ -> WorkspaceTreeReduction

let combine_family_for_execution = function
  | WorkspaceTreeReduction -> SharedTreeCombine
  | SingleBlockReduction | AtomicOutputReduction -> NoCombineKernel

let pipeline_launch_buckets optimizations reduction_family
    (metrics : Tensor_ir.body_metrics) (launch_buckets : launch_bucket list) =
  if
    Optimizations.enabled optimizations Optimizations.CudaReductionPipelinePlan
  then
    match (reduction_family, metrics.has_branch, metrics.has_div) with
    | ("ratio", _, true) | ("branchy", true, true) -> (
        match launch_buckets with
        | small_bucket :: medium_bucket :: rest ->
            { small_bucket with block_size = 256; num_warps = 4 }
            :: { medium_bucket with block_size = 256; num_warps = 4 }
               :: List.map
                    (fun (bucket : launch_bucket) ->
                      {
                        bucket with
                        block_size = min 512 (max 256 bucket.block_size);
                        num_warps = 4;
                      })
                    rest
        | _ -> launch_buckets)
    | (("robust" | "clipped-robust"), true, false) -> (
        match launch_buckets with
        | small_bucket :: medium_bucket :: rest ->
            { small_bucket with block_size = 256; num_warps = 4 }
            :: { medium_bucket with block_size = 256; num_warps = 4 }
               :: List.map
                    (fun (bucket : launch_bucket) ->
                      {
                        bucket with
                        block_size = min 512 (max 256 bucket.block_size);
                        num_warps = 4;
                      })
                    rest
        | _ -> launch_buckets)
    | _ -> launch_buckets
  else launch_buckets

let family_launch_buckets optimizations reduction_family
    (metrics : Tensor_ir.body_metrics) (launch_buckets : launch_bucket list) =
  let launch_buckets =
    pipeline_launch_buckets optimizations reduction_family metrics launch_buckets
  in
  match
    (reduction_family, metrics.has_branch, metrics.has_div, metrics.scalar_complexity)
  with
  | (("norm-square" | "affine-norm-square"), false, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = 512; num_warps = 8 }
          :: { medium_bucket with block_size = 512; num_warps = 8 }
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    { bucket with block_size = max bucket.block_size 1024; num_warps = 8 })
                  rest
      | _ -> launch_buckets)
  | ("dot-product", false, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = 512; num_warps = 8 }
          :: { medium_bucket with block_size = 256; num_warps = 8 }
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    { bucket with block_size = min bucket.block_size 512; num_warps = 8 })
                  rest
      | _ -> launch_buckets)
  | ("weighted-product", false, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = 256; num_warps = 4 }
          :: { medium_bucket with block_size = 512; num_warps = 8 }
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    { bucket with block_size = min bucket.block_size 512; num_warps = 8 })
                  rest
      | _ -> launch_buckets)
  | ("delta-square", false, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = 512; num_warps = 8 }
          :: { medium_bucket with block_size = 256; num_warps = 4 }
             :: (List.map
                   (fun (bucket : launch_bucket) ->
                     {
                       bucket with
                       block_size = min bucket.block_size 512;
                     num_warps = 8;
                    })
                  rest)
      | _ -> launch_buckets)
  | ("clipped-robust", true, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = 256; num_warps = 4 }
          :: { medium_bucket with block_size = 256; num_warps = 4 }
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    {
                      bucket with
                      block_size = min 512 (max 256 bucket.block_size);
                      num_warps = 4;
                    })
                  rest
      | _ -> launch_buckets)
  | ("branchy", true, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          { small_bucket with block_size = max 256 small_bucket.block_size; num_warps = 4 }
          :: { medium_bucket with block_size = max 256 medium_bucket.block_size; num_warps = 4 }
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    {
                      bucket with
                      block_size = min 512 (max 256 bucket.block_size);
                      num_warps = 4;
                    })
                  rest
      | _ -> launch_buckets)
  | ("mapped", false, false, scalar_complexity) when scalar_complexity <= 1 ->
      (match launch_buckets with
      | small_bucket :: rest ->
          { small_bucket with block_size = 512; num_warps = 8 } :: rest
      | [] -> launch_buckets)
  | (("mapped" | "weighted"), false, false, _) -> (
      match launch_buckets with
      | small_bucket :: medium_bucket :: rest ->
          small_bucket
          :: medium_bucket
             :: List.map
                  (fun (bucket : launch_bucket) ->
                    {
                      bucket with
                      block_size = min bucket.block_size 512;
                      num_warps = 8;
                    })
                  rest
      | _ -> launch_buckets)
  | _ -> launch_buckets

let assign_temp_slots steps =
  let steps_with_ids =
    steps
    |> List.mapi (fun index step ->
           let node_id, output =
             match step with
             | Elementwise step -> (step.node_id, step.output)
             | Reduction step -> (step.node_id, step.output)
           in
           (index, node_id, output))
  in
  let last_use =
    List.fold_left
      (fun acc (index, _, _) ->
        let inputs =
          match List.nth steps index with
          | Elementwise step -> List.map snd step.inputs
          | Reduction step -> (
              match step.source with
              | PlainInput input -> [ input ]
              | MappedInput { inputs; _ } -> List.map snd inputs)
        in
        List.fold_left
          (fun map input_name ->
            Kernel_plan_common.Int_map.add (Hashtbl.hash input_name) index map)
          acc inputs)
      Kernel_plan_common.Int_map.empty steps_with_ids
  in
  let next_slot = ref 0 in
  let live = ref [] in
  let slot_map = Hashtbl.create 32 in
  List.iteri
    (fun index step ->
      let still_live, freed =
        List.partition (fun (_, until) -> until >= index) !live
      in
      live := still_live;
      let output =
        match step with
        | Elementwise step -> step.output
        | Reduction step -> step.output
      in
      if not (String.equal output "out") then (
        let key = Hashtbl.hash output in
        let reusable =
          freed
          |> List.sort (fun (_, a) (_, b) -> compare a b)
          |> List.find_opt (fun _ -> true)
          |> Option.map fst
        in
        let slot =
          match reusable with
          | Some slot -> slot
          | None ->
              let slot = !next_slot in
              incr next_slot;
              slot
        in
        let until =
          Option.value
            (Kernel_plan_common.Int_map.find_opt key last_use)
            ~default:index
        in
        Hashtbl.replace slot_map output slot;
        live :=
          (slot, until)
          :: List.filter (fun (existing, _) -> existing <> slot) !live))
    steps;
  (Hashtbl.to_seq slot_map |> List.of_seq, !next_slot)

let rewrite_storage_names steps slot_map =
  let storage_name_of output =
    match List.assoc_opt output slot_map with
    | Some slot -> Printf.sprintf "tmp_slot_%d" slot
    | None -> output
  in
  List.map
    (function
      | Elementwise step ->
          let storage_class =
            match List.assoc_opt step.output slot_map with
            | Some slot -> TemporaryStorage slot
            | None -> OutputStorage
          in
          Elementwise
            {
              step with
              output = storage_name_of step.output;
              inputs =
                List.map
                  (fun (name, source) -> (name, storage_name_of source))
                  step.inputs;
              temp_slot =
                (match storage_class with
                | TemporaryStorage slot -> Some slot
                | OutputStorage -> None);
              storage_class;
            }
      | Reduction step ->
          let storage_class =
            match List.assoc_opt step.output slot_map with
            | Some slot -> TemporaryStorage slot
            | None -> OutputStorage
          in
          let source =
            match step.source with
            | PlainInput input -> PlainInput (storage_name_of input)
            | MappedInput { inputs; scalar_params; body } ->
                MappedInput
                  {
                    inputs =
                      List.map
                        (fun (name, source) -> (name, storage_name_of source))
                        inputs;
                    scalar_params;
                    body;
                  }
          in
          Reduction
            {
              step with
              output = storage_name_of step.output;
              source;
              temp_slot =
                (match storage_class with
                | TemporaryStorage slot -> Some slot
                | OutputStorage -> None);
              storage_class;
            })
    steps

let of_program ?(optimizations = Optimizations.none)
    (program : Tensor_ir.program) =
  let counts = Kernel_plan_common.use_counts program in
  let final_node = Kernel_plan_common.result_node_id program.result in
  let output_name_for_node id =
    match final_node with
    | Some final_id when final_id = id -> "out"
    | _ -> Printf.sprintf "tmp%d" id
  in
  let source_of_ref = function
    | Tensor_ir.ParamRef name -> name
    | Tensor_ir.NodeRef id -> output_name_for_node id
  in
  let steps =
    program.nodes
    |> List.filter_map (function
         | Tensor_ir.Elementwise1D
             { id; inputs; scalar_params; body; metrics; handling_hint; _ } ->
             let block_size, num_warps, launch_buckets, plan_class,
                 pointwise_class, complexity_bucket =
               Pointwise_plan.choose_elementwise_launch optimizations inputs metrics
             in
             let traits =
               if
                 Optimizations.enabled optimizations
                   Optimizations.CudaBodyTraitAnalysis
                 || Optimizations.enabled optimizations
                      Optimizations.SharedBodyTraitAnalysis
               then pointwise_traits (List.length inputs) scalar_params metrics body
               else []
              in
             let pointwise_family =
               pointwise_family_for_step optimizations program.entry_name pointwise_class
                 complexity_bucket (List.length inputs) metrics traits
             in
             Some
               (Elementwise
                  {
                    node_id = id;
                    kernel_name =
                      Printf.sprintf "%s_elementwise_%d" program.entry_name id;
                    output = output_name_for_node id;
                    inputs =
                      List.map
                        (fun (binding : Tensor_ir.input_binding) ->
                          (binding.name, source_of_ref binding.source))
                        inputs;
                    scalar_params;
                    block_size;
                    num_warps;
                    launch_buckets;
                    plan_class;
                    pointwise_class;
                    pointwise_family;
                    traits;
                    complexity_bucket;
                    producer_strategy =
                      Kernel_plan_common.producer_strategy_of_hint handling_hint;
                    storage_class = OutputStorage;
                    temp_slot = None;
                  })
         | Tensor_ir.Reduce1D
             {
               id;
               source = tensor_source;
               kind;
               metrics;
               reduction_shape;
               handling_hint;
               _;
             } ->
             let block_size, num_warps, launch_buckets, single_block_threshold,
                 small_reduction_threshold, small_program_count,
                 reduction_family, reduction_class, reduction_strategy,
                 strategy_kind, stage_count, stage_layout, complexity_bucket =
               Reduction_plan.choose_reduction_launch optimizations reduction_shape
                 metrics kind
             in
             let reduction_family =
               backend_reduction_family optimizations reduction_family metrics
                 tensor_source
               |> selected_reduction_family optimizations program.entry_name
             in
             let launch_buckets =
               family_launch_buckets optimizations reduction_family metrics
                 launch_buckets
             in
             let traits =
               if
                 Optimizations.enabled optimizations
                   Optimizations.CudaBodyTraitAnalysis
                 || Optimizations.enabled optimizations
                      Optimizations.SharedBodyTraitAnalysis
               then reduction_traits reduction_family tensor_source metrics
               else []
             in
             let use_shuffle_reduction =
               Optimizations.enabled optimizations
                 Optimizations.CudaReductionShuffle
               && kind = Tensor_ir.Sum
               && (List.mem reduction_family
                     [
                       "mapped";
                       "norm-square";
                       "affine-norm-square";
                       "dot-product";
                       "delta-square";
                       "ratio";
                       "branchy";
                       "robust";
                       "clipped-robust";
                     ]
                  || (Optimizations.enabled optimizations
                        Optimizations.CudaReductionInstructionSelect
                     && String.equal reduction_family "weighted-product"))
             in
             let traits =
               traits
               |> add_trait use_shuffle_reduction "shuffle-reduce"
               |> unique_traits
             in
             let source =
               match tensor_source with
               | Tensor_ir.PlainInput (Tensor_ir.NodeRef producer_id)
                 when
                   Optimizations.enabled optimizations
                     Optimizations.ReductionPrecombine -> (
                   match
                     ( Kernel_plan_common.Int_map.find_opt producer_id counts,
                       Kernel_plan_common.find_node program producer_id )
                   with
                   | ( Some 1,
                       Tensor_ir.Elementwise1D
                         { inputs; scalar_params; body; _ } )
                     when
                       Kernel_plan_common.scalar_expr_complexity body
                       <= optimizations.config
                            .reduction_precombine_max_body_complexity ->
                       let producer_metrics =
                         Tensor_ir.
                           {
                             scalar_complexity =
                               Kernel_plan_common.scalar_expr_complexity body;
                             has_branch =
                               Kernel_plan_common.scalar_expr_has_branch body;
                             has_div =
                               Kernel_plan_common.scalar_expr_has_div body;
                             branch_count = 0;
                             div_count = 0;
                             repeated_subexpressions = 0;
                             estimated_uses = 1;
                           }
                       in
                       if
                         Optimizations.enabled optimizations
                           Optimizations.ReductionLateFusionGuard
                         && producer_metrics.scalar_complexity
                            >= optimizations.config
                                 .reduction_late_fusion_complexity_threshold
                       then PlainInput (source_of_ref (Tensor_ir.NodeRef producer_id))
                       else
                         MappedInput
                           {
                             inputs =
                               List.map
                                 (fun (binding : Tensor_ir.input_binding) ->
                                   (binding.name, source_of_ref binding.source))
                                 inputs;
                             scalar_params;
                             body;
                           }
                   | _ ->
                       let input =
                         match tensor_source with
                         | Tensor_ir.PlainInput input -> input
                         | _ -> assert false
                       in
                       PlainInput (source_of_ref input))
               | Tensor_ir.PlainInput input -> PlainInput (source_of_ref input)
               | MappedInput { inputs; scalar_params; body } ->
                   let inputs =
                     List.map
                       (fun (binding : Tensor_ir.input_binding) ->
                         (binding.name, source_of_ref binding.source))
                       inputs
                   in
                   MappedInput { inputs; scalar_params; body }
             in
            let execution_family =
               reduction_execution_for_step optimizations reduction_family complexity_bucket
                 metrics kind reduction_strategy
             in
             Some
               (Reduction
                  {
                    node_id = id;
                    kernel_name =
                      Printf.sprintf "%s_reduce_%d" program.entry_name id;
                    combine_kernel_name =
                      Printf.sprintf "%s_reduce_%d_combine" program.entry_name id;
                    output = output_name_for_node id;
                    reduce_kind = kind;
                    block_size;
                    num_warps;
                    launch_buckets;
                    single_block_threshold;
                    small_reduction_threshold;
                    small_program_count;
                    source;
                    reduction_family;
                    reduction_class;
                    strategy_kind;
                    reduction_strategy;
                    execution_family;
                    combine_family =
                      combine_family_for_execution execution_family;
                    traits;
                    uses_workspace =
                      execution_family = WorkspaceTreeReduction;
                    stage_count;
                    stage_layout =
                      (if use_shuffle_reduction then "cuda-shuffle"
                       else stage_layout);
                    complexity_bucket;
                    producer_strategy =
                      Kernel_plan_common.producer_strategy_of_hint handling_hint;
                    storage_class = OutputStorage;
                    temp_slot = None;
                  }))
  in
  let slot_map, slot_count =
    if
      Optimizations.enabled optimizations Optimizations.TempLifetimePack
      || Optimizations.enabled optimizations Optimizations.StorageReusePack
    then assign_temp_slots steps
    else ([], 0)
  in
  let steps =
    if
      Optimizations.enabled optimizations Optimizations.StorageReusePack
      || Optimizations.enabled optimizations Optimizations.TempLifetimePack
    then rewrite_storage_names steps slot_map
    else steps
  in
  let result_name =
    match program.result with
    | Tensor_ir.TensorResult (Tensor_ir.ParamRef name)
    | Tensor_ir.ScalarResult (Tensor_ir.ParamRef name) ->
        name
    | Tensor_ir.TensorResult (Tensor_ir.NodeRef _)
    | Tensor_ir.ScalarResult (Tensor_ir.NodeRef _) ->
        "out"
  in
  let temporary_count =
    if
      Optimizations.enabled optimizations Optimizations.TempLifetimePack
      || Optimizations.enabled optimizations Optimizations.StorageReusePack
    then slot_count
    else
      List.fold_left
        (fun count -> function
          | Elementwise { output; _ } | Reduction { output; _ } ->
              if String.equal output "out" then count else count + 1)
        0 steps
  in
  { entry_name = program.entry_name; steps; result_name; temporary_count }

let step_to_yojson = function
  | Elementwise
      {
        node_id;
        kernel_name;
        output;
        inputs;
        scalar_params;
        block_size;
        num_warps;
        launch_buckets;
        plan_class;
        pointwise_class;
        pointwise_family;
        traits;
        complexity_bucket;
        producer_strategy;
        storage_class;
        temp_slot;
      } ->
      `Assoc
        [
          ("kind", `String "elementwise");
          ("node_id", `Int node_id);
          ("kernel_name", `String kernel_name);
          ("output", `String output);
          ( "inputs",
            `List
              (List.map
                 (fun (name, source) ->
                   `Assoc [ ("name", `String name); ("source", `String source) ])
                 inputs) );
          ( "scalar_params",
            `List (List.map (fun name -> `String name) scalar_params) );
          ("block_size", `Int block_size);
          ("num_warps", `Int num_warps);
          ("plan_class", `String plan_class);
          ("pointwise_class", `String (pointwise_class_to_string pointwise_class));
          ("pointwise_family", `String (pointwise_family_to_string pointwise_family));
          ("traits", traits_to_yojson traits);
          ("complexity_bucket", complexity_bucket_to_yojson complexity_bucket);
          ("producer_strategy", `String (producer_strategy_to_string producer_strategy));
          ("storage_class", storage_class_to_yojson storage_class);
          ("temp_slot", match temp_slot with Some slot -> `Int slot | None -> `Null);
          ("launch_buckets", `List (List.map launch_bucket_to_yojson launch_buckets));
        ]
  | Reduction
      {
        node_id;
        kernel_name;
        combine_kernel_name;
        output;
        reduce_kind;
        block_size;
        num_warps;
        launch_buckets;
        single_block_threshold;
        small_reduction_threshold;
        small_program_count;
        source;
        reduction_family;
        reduction_class;
        strategy_kind;
        reduction_strategy;
        execution_family;
        combine_family;
        traits;
        uses_workspace;
        stage_count;
        stage_layout;
        complexity_bucket;
        producer_strategy;
        storage_class;
        temp_slot;
      } ->
      `Assoc
        [
          ("kind", `String "reduction");
          ("node_id", `Int node_id);
          ("kernel_name", `String kernel_name);
          ("combine_kernel_name", `String combine_kernel_name);
          ("output", `String output);
          ("reduce_kind", `String (reduce_kind_to_string reduce_kind));
          ("block_size", `Int block_size);
          ("num_warps", `Int num_warps);
          ("launch_buckets", `List (List.map launch_bucket_to_yojson launch_buckets));
          ( "single_block_threshold",
            match single_block_threshold with Some value -> `Int value | None -> `Null );
          ( "small_reduction_threshold",
            match small_reduction_threshold with Some value -> `Int value | None -> `Null );
          ( "small_program_count",
            match small_program_count with Some value -> `Int value | None -> `Null );
          ("source", reduction_source_to_yojson source);
          ("reduction_family", `String reduction_family);
          ("reduction_class", `String reduction_class);
          ("strategy_kind", `String strategy_kind);
          ("reduction_strategy", `String (reduction_strategy_to_string reduction_strategy));
          ("execution_family", `String (reduction_execution_to_string execution_family));
          ("combine_family", `String (combine_family_to_string combine_family));
          ("traits", traits_to_yojson traits);
          ("uses_workspace", `Bool uses_workspace);
          ("stage_count", `Int stage_count);
          ("stage_layout", `String stage_layout);
          ("complexity_bucket", complexity_bucket_to_yojson complexity_bucket);
          ("producer_strategy", `String (producer_strategy_to_string producer_strategy));
          ("storage_class", storage_class_to_yojson storage_class);
          ("temp_slot", match temp_slot with Some slot -> `Int slot | None -> `Null);
        ]

let to_yojson { entry_name; steps; result_name; temporary_count } =
  `Assoc
    [
      ("entry_name", `String entry_name);
      ("steps", `List (List.map step_to_yojson steps));
      ("result_name", `String result_name);
      ("temporary_count", `Int temporary_count);
    ]

let to_string plan = Yojson.Safe.pretty_to_string (to_yojson plan)

let source_name_of_ref _plan = function
  | Tensor_ir.ParamRef name -> name
  | Tensor_ir.NodeRef id -> Printf.sprintf "tmp%d" id
