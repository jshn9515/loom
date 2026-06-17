open Kernel_plan_common

let choose_elementwise_launch optimizations inputs
    (metrics : Tensor_ir.body_metrics) =
  let buckets =
    if
      Optimizations.enabled optimizations Optimizations.LaunchBucketSpecialize
      || Optimizations.enabled optimizations
           Optimizations.CudaPointwiseSmallShapePlan
    then launch_buckets_from_config optimizations.config.elementwise_buckets
    else []
  in
  let complexity = metrics.scalar_complexity in
  let complexity_bucket = complexity_bucket_of_score complexity in
  let block_size, num_warps =
    if
      Optimizations.enabled optimizations
        Optimizations.CudaPointwiseSmallShapePlan
    then
      let input_count = List.length inputs in
      match
        (complexity_bucket, input_count, metrics.has_branch, metrics.has_div)
      with
      | (Tiny | Small), 1, true, _ -> (256, 4)
      | Tiny, 1, _, _ -> (512, 8)
      | Small, 1, _, _ -> (384, 8)
      | (Tiny | Small), _, _, _ when input_count <= 2 -> (512, 8)
      | Medium, 1, true, _ -> (256, 4)
      | Medium, _, _, _ -> (256, 4)
      | _ -> (256, 4)
    else if Optimizations.enabled optimizations Optimizations.PointwiseShapePlan then
      let input_count = List.length inputs in
      match
        (complexity_bucket, input_count, metrics.has_branch, metrics.has_div)
      with
      | _, _, true, _ -> (128, 4)
      | _, _, _, true -> (256, 4)
      | Tiny, 1, _, _ -> (1024, 8)
      | Tiny, _, _, _ -> (512, 8)
      | Small, _, _, _ when input_count <= 2 -> (512, 8)
      | Medium, _, _, _ -> (256, 4)
      | Large, _, _, _ -> (128, 4)
      | _ -> (256, 4)
    else if
      Optimizations.enabled optimizations Optimizations.ElementwisePlanSpecialize
    then
      let input_count = List.length inputs in
      if input_count <= 1 && complexity <= 2 then (512, 8)
      else if input_count <= 2 && complexity <= 5 then (512, 8)
      else (256, 4)
    else (default_block_size, default_num_warps)
  in
  let pointwise_class =
    if
      (not metrics.has_branch) && not metrics.has_div
      && List.length inputs <= 2
    then PointwiseFastPath
    else GeneralPointwise
  in
  let plan_class =
    if metrics.has_branch then "branchy-pointwise"
    else if List.length inputs > 1 then "multi-input-pointwise"
    else "simple-pointwise"
  in
  (block_size, num_warps, buckets, plan_class, pointwise_class, complexity_bucket)
