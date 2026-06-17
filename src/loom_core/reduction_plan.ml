open Kernel_plan_common

let classify_reduction_shape = function
  | Tensor_ir.PlainReduction -> "plain"
  | MappedReduction -> "mapped"
  | WeightedMappedReduction -> "weighted-mapped"
  | BranchyMappedReduction -> "branchy-mapped"
  | RatioMappedReduction -> "ratio-mapped"

let choose_reduction_launch optimizations reduction_shape
    (metrics : Tensor_ir.body_metrics) kind =
  let reduction_class =
    if Optimizations.enabled optimizations Optimizations.ReductionClassify then
      classify_reduction_shape reduction_shape
    else "plain"
  in
  let body_score = reduction_body_score metrics in
  let complexity_bucket = complexity_bucket_of_score body_score in
  let reduction_family =
    match (reduction_class, metrics.has_branch, metrics.has_div, kind) with
    | "plain", false, false, Tensor_ir.Sum -> "plain-sum"
    | "plain", false, false, Tensor_ir.MaxReduce -> "plain-max"
    | "weighted-mapped", _, _, _ -> "weighted"
    | "ratio-mapped", _, true, _ -> "ratio"
    | "branchy-mapped", true, _, _ -> "branchy"
    | "mapped", _, _, _ when metrics.repeated_subexpressions > 0 -> "mapped-reuse"
    | "mapped", _, _, _ -> "mapped"
    | _ -> reduction_class
  in
  let base_buckets =
    if Optimizations.enabled optimizations Optimizations.LaunchBucketSpecialize
    then launch_buckets_from_config optimizations.config.reduction_buckets
    else []
  in
  let adjust_bucket (bucket : launch_bucket) =
    if
      not
        (Optimizations.enabled optimizations
           Optimizations.ReductionClassAwareBuckets)
    then bucket
    else
      match reduction_family with
      | "branchy" | "ratio" ->
          {
            bucket with
            block_size = max 128 (bucket.block_size / 2);
            num_warps = min bucket.num_warps 4;
          }
      | "weighted" | "mapped-reuse" ->
          if bucket.max_n <= 131072 then
            { bucket with block_size = max 256 bucket.block_size; num_warps = max 4 bucket.num_warps }
          else if bucket.max_n <= 2097152 then
            { bucket with block_size = max 256 bucket.block_size; num_warps = max 4 bucket.num_warps }
          else
            { bucket with block_size = max 512 bucket.block_size; num_warps = max 8 bucket.num_warps }
      | "plain-sum" ->
          { bucket with block_size = min 1024 (max bucket.block_size 256) }
      | _ -> bucket
  in
  let buckets = List.map adjust_bucket base_buckets in
  let default_launch =
    if Optimizations.enabled optimizations Optimizations.ReductionTreePlan then
      match (reduction_family, complexity_bucket, kind) with
      | "branchy", _, _ -> (256, 4)
      | ("ratio" | "weighted" | "mapped-reuse"), Large, _ -> (256, 4)
      | ("ratio" | "weighted" | "mapped-reuse"), _, _ -> (512, 4)
      | "mapped", Large, _ -> (256, 4)
      | _, _, Tensor_ir.MaxReduce -> (512, 8)
      | _ -> (1024, 8)
    else if Optimizations.enabled optimizations Optimizations.ReductionPlanSpecialize
    then
      match kind with
      | Tensor_ir.Sum -> (1024, 8)
      | Tensor_ir.MaxReduce -> (512, 8)
    else (default_block_size, default_num_warps)
  in
  let block_size, num_warps =
    if Optimizations.enabled optimizations Optimizations.ReductionPartialShapePack
    then
      match reduction_family with
      | "plain-sum" when body_score <= 2 -> default_launch
      | "branchy" | "ratio" ->
          (max 128 (fst default_launch / 2), min 4 (snd default_launch))
      | _
        when
          body_score
          >= optimizations.config.reduction_stage_large_threshold / 32768 ->
          (max 256 (fst default_launch / 2), min 4 (snd default_launch))
      | _ -> default_launch
    else default_launch
  in
  let single_block_threshold =
    if
      Optimizations.enabled optimizations Optimizations.ReductionSplitPlan
      || Optimizations.enabled optimizations Optimizations.ReductionTwoPhase
      || Optimizations.enabled optimizations Optimizations.ReductionStageSizing
      || Optimizations.enabled optimizations Optimizations.ReductionStageBalance
    then
      let threshold = optimizations.config.single_block_reduction_threshold in
      let adjusted =
        if Optimizations.enabled optimizations Optimizations.ReductionStageBalance
        then
          match reduction_family with
          | "branchy" | "ratio" -> max 128 (threshold / 2)
          | "weighted" | "mapped-reuse" -> threshold
          | "plain-sum" when body_score <= 2 -> threshold * 2
          | _ -> threshold
        else threshold
      in
      Some adjusted
    else None
  in
  let small_reduction_threshold, small_program_count, small_direct =
    match reduction_family with
    | "branchy"
      when
        kind = Tensor_ir.Sum
        && Optimizations.enabled optimizations Optimizations.BranchyReductionRescue
        && (complexity_bucket = Small || complexity_bucket = Medium) ->
        ( Some optimizations.config.branchy_small_reduction_threshold,
          Some optimizations.config.branchy_small_reduction_program_count,
          false )
    | _
      when
        Optimizations.enabled optimizations
          Optimizations.ReductionSmallPlanSpecialize ->
        begin
          match reduction_family with
          | "plain-sum"
            when kind = Tensor_ir.Sum
                 && not metrics.has_branch
                 && not metrics.has_div
                 && (complexity_bucket = Tiny || complexity_bucket = Small) ->
              ( Some optimizations.config.small_direct_reduction_threshold,
                None,
                true )
          | "mapped"
            when kind = Tensor_ir.Sum
                 && not metrics.has_branch
                 && not metrics.has_div
                 && (complexity_bucket = Tiny || complexity_bucket = Small) ->
              ( Some optimizations.config.small_partial_reduction_threshold,
                Some optimizations.config.small_reduction_program_count,
                false )
          | ("weighted" | "mapped-reuse")
            when kind = Tensor_ir.Sum
                 && not metrics.has_branch
                 && not metrics.has_div
                 && (complexity_bucket = Tiny || complexity_bucket = Small) ->
              ( Some optimizations.config.small_partial_reduction_threshold,
                Some (max 4 (optimizations.config.small_reduction_program_count / 2)),
                false
              )
          | _ -> (None, None, false)
        end
    | _ -> (None, None, false)
  in
  let stage_count =
    if small_reduction_threshold <> None && small_direct then 1
    else if small_reduction_threshold <> None then 2
    else if Optimizations.enabled optimizations Optimizations.ReductionStageBalance then
      if
        body_score >= 12 || reduction_family = "branchy"
        || reduction_family = "ratio"
      then 3
      else if
        body_score >= 6 || reduction_family = "weighted"
        || reduction_family = "mapped-reuse"
      then 2
      else 1
    else if Optimizations.enabled optimizations Optimizations.ReductionStageSizing then
      if body_score >= 12 then 3
      else if
        body_score >= 5 || reduction_class = "branchy-mapped"
        || reduction_class = "ratio-mapped"
      then 2
      else 1
    else if Optimizations.enabled optimizations Optimizations.ReductionSplitPlan then
      match complexity_bucket with
      | Tiny | Small -> 1
      | Medium -> 2
      | Large -> 3
    else if Optimizations.enabled optimizations Optimizations.ReductionTwoPhase
    then 2
    else 1
  in
  let reduction_strategy =
    if small_reduction_threshold <> None && small_direct then SmallDirectReduction
    else if small_reduction_threshold <> None then SmallPartialReduction
    else if
      Optimizations.enabled optimizations Optimizations.ReductionStageSizing
      || Optimizations.enabled optimizations Optimizations.ReductionSplitPlan
    then
      if stage_count = 1 then DirectReduction
      else if stage_count = 2 then SingleStagePartialReduction
      else MultiStageTreeReduction
    else if Optimizations.enabled optimizations Optimizations.ReductionTwoPhase
    then TwoPhaseThresholdedReduction
    else FixedStrategy
  in
  let strategy_kind = reduction_strategy_to_string reduction_strategy in
  let stage_layout =
    if reduction_strategy = SmallDirectReduction then "small-direct"
    else if reduction_strategy = SmallPartialReduction then "small-partial+combine"
    else if stage_count <= 1 then "single"
    else if stage_count = 2 then "partial+combine"
    else "multi-stage-tree"
  in
  ( block_size,
    num_warps,
    buckets,
    single_block_threshold,
    small_reduction_threshold,
    small_program_count,
    reduction_family,
    reduction_class,
    reduction_strategy,
    strategy_kind,
    stage_count,
    stage_layout,
    complexity_bucket )
