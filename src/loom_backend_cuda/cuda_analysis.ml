let classify_result = function
  | Tensor_ir.TensorResult _ -> "tensor"
  | Tensor_ir.ScalarResult _ -> "scalar"

let traits_to_yojson traits =
  `List (List.map (fun name -> `String name) traits)

let step_summary = function
  | Cuda_plan.Elementwise step ->
      `Assoc
        [
          ("kind", `String "elementwise");
          ("node_id", `Int step.node_id);
          ("kernel_name", `String step.kernel_name);
          ("plan_class", `String step.plan_class);
          ("pointwise_class", `String (Cuda_plan.pointwise_class_to_string step.pointwise_class));
          ("pointwise_family", `String (Cuda_plan.pointwise_family_to_string step.pointwise_family));
          ("traits", traits_to_yojson step.traits);
          ("complexity_bucket", Cuda_plan.complexity_bucket_to_yojson step.complexity_bucket);
          ("producer_strategy", `String (Cuda_plan.producer_strategy_to_string step.producer_strategy));
          ("block_size", `Int step.block_size);
          ("num_warps", `Int step.num_warps);
        ]
  | Cuda_plan.Reduction step ->
      `Assoc
        [
          ("kind", `String "reduction");
          ("node_id", `Int step.node_id);
          ("kernel_name", `String step.kernel_name);
          ("combine_kernel_name", `String step.combine_kernel_name);
          ("reduction_family", `String step.reduction_family);
          ("reduction_class", `String step.reduction_class);
          ("traits", traits_to_yojson step.traits);
          ("strategy_kind", `String step.strategy_kind);
          ("execution_family", `String (Cuda_plan.reduction_execution_to_string step.execution_family));
          ("combine_family", `String (Cuda_plan.combine_family_to_string step.combine_family));
          ("stage_layout", `String step.stage_layout);
          ("stage_count", `Int step.stage_count);
          ("complexity_bucket", Cuda_plan.complexity_bucket_to_yojson step.complexity_bucket);
          ("producer_strategy", `String (Cuda_plan.producer_strategy_to_string step.producer_strategy));
          ("block_size", `Int step.block_size);
          ("num_warps", `Int step.num_warps);
          ("uses_shared_workspace", `Bool step.uses_workspace);
        ]

let to_yojson ~(program : Tensor_ir.program) ~(plan : Cuda_plan.t) =
  let reduction_count, elementwise_count =
    List.fold_left
      (fun (red, elt) -> function
        | Cuda_plan.Reduction _ -> (red + 1, elt)
        | Cuda_plan.Elementwise _ -> (red, elt + 1))
      (0, 0) plan.steps
  in
  `Assoc
    [
      ("backend", `String "cuda");
      ("entry_name", `String plan.entry_name);
      ("result_kind", `String (classify_result program.result));
      ("step_count", `Int (List.length plan.steps));
      ("elementwise_step_count", `Int elementwise_count);
      ("reduction_step_count", `Int reduction_count);
      ("temporary_count", `Int plan.temporary_count);
      ("steps", `List (List.map step_summary plan.steps));
    ]

let to_string ~program ~plan =
  Yojson.Safe.pretty_to_string (to_yojson ~program ~plan)
