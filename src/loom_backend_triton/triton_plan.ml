include Kernel_plan_common

let triton_family_launch_buckets optimizations reduction_family
    (metrics : Tensor_ir.body_metrics) (launch_buckets : launch_bucket list) =
  if
    not
      (Optimizations.enabled optimizations
         Optimizations.TritonReductionInstructionSelect)
  then launch_buckets
  else
    match
      ( reduction_family,
        metrics.has_branch,
        metrics.has_div,
        metrics.scalar_complexity )
    with
    | ( ( "norm-square" | "affine-norm-square" | "dot-product"
        | "weighted-product" | "delta-square" ),
        false,
        false,
        _ ) -> (
        match launch_buckets with
        | small_bucket :: medium_bucket :: rest ->
            let small_block_size =
              if
                Optimizations.enabled optimizations
                  Optimizations.TritonSmallReductionTailTune
              then 512
              else 1024
            in
            { small_bucket with block_size = small_block_size; num_warps = 8 }
            :: { medium_bucket with block_size = 512; num_warps = 8 }
               :: List.map
                    (fun (bucket : launch_bucket) ->
                      {
                        bucket with
                        block_size = min 1024 (max 512 bucket.block_size);
                        num_warps = 8;
                      })
                    rest
        | _ -> launch_buckets)
    | (("ratio" | "branchy" | "robust" | "clipped-robust"), _, _, _) -> (
        match launch_buckets with
        | small_bucket :: medium_bucket :: rest ->
            { small_bucket with block_size = 512; num_warps = 4 }
            :: { medium_bucket with block_size = 512; num_warps = 4 }
               :: List.map
                    (fun (bucket : launch_bucket) ->
                      {
                        bucket with
                        block_size = min 1024 (max 512 bucket.block_size);
                        num_warps = 4;
                      })
                    rest
        | _ -> launch_buckets)
    | ("mapped", false, false, scalar_complexity) when scalar_complexity <= 1 -> (
        match launch_buckets with
        | small_bucket :: rest ->
            { small_bucket with block_size = 1024; num_warps = 8 } :: rest
        | [] -> launch_buckets)
    | _ -> launch_buckets

let of_program ?(optimizations = Optimizations.none)
    (program : Tensor_ir.program) =
  let counts = use_counts program in
  let final_node = result_node_id program.result in
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
                   Optimizations.SharedBodyTraitAnalysis
                 || Optimizations.enabled optimizations
                      Optimizations.TritonPointwiseInstructionSelect
               then pointwise_traits (List.length inputs) scalar_params metrics body
               else []
             in
             let launch_buckets =
               if
                 Optimizations.enabled optimizations
                 Optimizations.TritonPointwiseInstructionSelect
                 && (has_trait "activation-or-threshold" traits
                    || has_trait "clip" traits)
               then
                 match launch_buckets with
                 | small_bucket :: rest ->
                     { small_bucket with block_size = 1024; num_warps = 4 }
                     :: rest
                 | [] -> launch_buckets
               else launch_buckets
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
                    traits;
                    complexity_bucket;
                    producer_strategy = producer_strategy_of_hint handling_hint;
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
               if
                 Optimizations.enabled optimizations
                   Optimizations.TritonReductionInstructionSelect
               then
                 specialized_reduction_family ~enable_norm:true ~enable_dot:true
                   ~enable_weighted:true reduction_family metrics tensor_source
               else reduction_family
             in
             let launch_buckets =
               triton_family_launch_buckets optimizations reduction_family metrics
                 launch_buckets
             in
             let traits =
               if
                 Optimizations.enabled optimizations
                   Optimizations.SharedBodyTraitAnalysis
                 || Optimizations.enabled optimizations
                      Optimizations.TritonReductionInstructionSelect
               then reduction_traits reduction_family tensor_source metrics
               else []
             in
             let source =
               match tensor_source with
               | Tensor_ir.PlainInput (Tensor_ir.NodeRef producer_id)
                 when
                   Optimizations.enabled optimizations
                     Optimizations.ReductionPrecombine -> (
                   match
                     ( Int_map.find_opt producer_id counts,
                       find_node program producer_id )
                   with
                   | ( Some 1,
                       Tensor_ir.Elementwise1D
                         { inputs; scalar_params; body; _ } )
                     when
                       scalar_expr_complexity body
                       <= optimizations.config
                            .reduction_precombine_max_body_complexity ->
                       let producer_metrics =
                         Tensor_ir.
                           {
                             scalar_complexity = scalar_expr_complexity body;
                             has_branch = scalar_expr_has_branch body;
                             has_div = scalar_expr_has_div body;
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
                    traits;
                    strategy_kind;
                    reduction_strategy;
                    stage_count;
                    stage_layout;
                    complexity_bucket;
                    producer_strategy = producer_strategy_of_hint handling_hint;
                    storage_class = OutputStorage;
                    temp_slot = None;
                  }))
  in
  let slot_map, slot_count =
    if
      Optimizations.enabled optimizations Optimizations.TempLifetimePack
      || Optimizations.enabled optimizations Optimizations.StorageReusePack
    then Storage_pack.assign_temp_slots steps
    else ([], 0)
  in
  let steps =
    if
      Optimizations.enabled optimizations Optimizations.StorageReusePack
      || Optimizations.enabled optimizations Optimizations.TempLifetimePack
    then Storage_pack.rewrite_storage_names steps slot_map
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
        traits;
        strategy_kind;
        reduction_strategy;
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
          ("traits", traits_to_yojson traits);
          ("strategy_kind", `String strategy_kind);
          ("reduction_strategy", `String (reduction_strategy_to_string reduction_strategy));
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
