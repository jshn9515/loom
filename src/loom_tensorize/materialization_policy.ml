let default_handling_hint = Tensor_ir.NoPreference

let materialization_hint optimizations metrics reduction_shape_opt =
  if not (Optimizations.enabled optimizations Optimizations.MaterializationChoice)
  then default_handling_hint
  else
    match reduction_shape_opt with
    | Some reduction_shape
      when
        Optimizations.enabled optimizations
          Optimizations.ReductionMaterializationChoice -> (
        match reduction_shape with
        | Tensor_ir.BranchyMappedReduction -> Tensor_ir.PreferMaterialize
        | Tensor_ir.RatioMappedReduction
          when
            metrics.Tensor_ir.scalar_complexity
            >= optimizations.config.reduction_materialize_complexity_threshold
            || metrics.Tensor_ir.div_count > 0 ->
            Tensor_ir.PreferMaterialize
        | Tensor_ir.WeightedMappedReduction
          when
            metrics.Tensor_ir.repeated_subexpressions > 0
            || metrics.Tensor_ir.estimated_uses > 1
          ->
            Tensor_ir.PreferClone
        | _ when metrics.Tensor_ir.estimated_uses <= 1 -> Tensor_ir.PreferFuse
        | _
          when
            metrics.Tensor_ir.scalar_complexity
            >= optimizations.config.reduction_materialize_complexity_threshold
            || metrics.Tensor_ir.repeated_subexpressions > 0 ->
            Tensor_ir.PreferMaterialize
        | _ -> Tensor_ir.PreferClone)
    | Some Tensor_ir.BranchyMappedReduction -> Tensor_ir.PreferMaterialize
    | Some Tensor_ir.RatioMappedReduction
      when metrics.Tensor_ir.scalar_complexity >= 3 ->
        Tensor_ir.PreferMaterialize
    | _ when metrics.Tensor_ir.estimated_uses <= 1 -> Tensor_ir.PreferFuse
    | _
      when metrics.Tensor_ir.scalar_complexity <= optimizations.config.small_clone_nodes ->
        Tensor_ir.PreferClone
    | _
      when
        metrics.Tensor_ir.has_branch
        || metrics.Tensor_ir.scalar_complexity
           >= optimizations.config.materialize_multi_use_complexity_threshold ->
        Tensor_ir.PreferMaterialize
    | _ -> Tensor_ir.PreferClone

let allow_inline_producer optimizations metrics use_count =
  let branch_guard =
    Optimizations.enabled optimizations Optimizations.BranchAwareFusionGuard
    && metrics.Tensor_ir.has_branch
    && metrics.Tensor_ir.scalar_complexity
       >= optimizations.config.branchy_fusion_complexity_threshold
  in
  let pointwise_guard =
    Optimizations.enabled optimizations Optimizations.PointwiseMaterializationGuard
    &&
    (metrics.Tensor_ir.has_branch || metrics.Tensor_ir.has_div)
    && metrics.Tensor_ir.scalar_complexity
       >= optimizations.config.pointwise_materialization_complexity_threshold
  in
  if branch_guard || pointwise_guard then `Disallow
  else if use_count <= 1 then `Fuse
  else if
    Optimizations.enabled optimizations Optimizations.MaterializationChoice
    && metrics.Tensor_ir.scalar_complexity
       >= optimizations.config.materialize_multi_use_complexity_threshold
  then `Disallow
  else `Clone

let allow_inline_into_reduction optimizations metrics use_count target_complexity =
  let late_guard =
    Optimizations.enabled optimizations Optimizations.ReductionLateFusionGuard
    &&
    (metrics.Tensor_ir.scalar_complexity + target_complexity
     >= optimizations.config.reduction_late_fusion_complexity_threshold
    || (metrics.Tensor_ir.has_branch && target_complexity > 0)
    || metrics.Tensor_ir.repeated_subexpressions
       >= optimizations.config.reduction_body_cse_min_occurrences)
  in
  if late_guard then `Disallow
  else if
    Optimizations.enabled optimizations
      Optimizations.ReductionMaterializationChoice
    &&
    (metrics.Tensor_ir.scalar_complexity
     >= optimizations.config.reduction_materialize_complexity_threshold
    || metrics.Tensor_ir.has_branch
    || metrics.Tensor_ir.div_count > 0)
    && use_count > 1
  then `Disallow
  else allow_inline_producer optimizations metrics use_count
