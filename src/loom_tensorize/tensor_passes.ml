open Tensor_ir

module Int_map = Map.Make (Int)
module Int_set = Set.Make (Int)

let canonicalize_program (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Tensor_ir.Elementwise1D ({ scalar_params; body; _ } as node) ->
            Elementwise1D
              {
                node with
                scalar_params = List.sort_uniq String.compare scalar_params;
                body = Reduction_analysis.canonicalize_scalar_expr body;
              }
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            Reduce1D
              {
                node with
                source =
                  MappedInput
                    {
                      mapped with
                      scalar_params =
                        List.sort_uniq String.compare mapped.scalar_params;
                      body =
                        Reduction_analysis.canonicalize_scalar_expr mapped.body;
                    };
              }
        | Reduce1D _ as node -> node)
      program.Tensor_ir.nodes
  in
  { program with nodes }

let simplify_program (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Tensor_ir.Elementwise1D ({ body; _ } as node) ->
            Elementwise1D
              {
                node with
                body = Reduction_analysis.simplify_scalar_expr body;
              }
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            Reduce1D
              {
                node with
                source =
                  MappedInput
                    {
                      mapped with
                      body =
                        Reduction_analysis.simplify_scalar_expr mapped.body;
                    };
              }
        | Reduce1D _ as node -> node)
      program.Tensor_ir.nodes
  in
  { program with nodes }

let reachable_node_ids (program : Tensor_ir.program) =
  let rec loop seen = function
    | [] -> seen
    | Tensor_ir.ParamRef _ :: rest -> loop seen rest
    | Tensor_ir.NodeRef id :: rest when Int_set.mem id seen -> loop seen rest
    | Tensor_ir.NodeRef id :: rest ->
        let seen = Int_set.add id seen in
        let next =
          match
            List.find_opt
              (function
                | Tensor_ir.Elementwise1D { id = node_id; _ }
                | Reduce1D { id = node_id; _ } ->
                    node_id = id)
              program.Tensor_ir.nodes
          with
          | Some (Tensor_ir.Elementwise1D { inputs; _ }) ->
              List.map
                (fun (binding : Tensor_ir.input_binding) -> binding.source)
                inputs
          | Some (Reduce1D { source; _ }) ->
              Reduction_analysis.reduction_sources source
          | None -> []
        in
        loop seen (next @ rest)
  in
  let roots =
    match program.Tensor_ir.result with
    | TensorResult value | ScalarResult value -> [ value ]
  in
  loop Int_set.empty roots

let shape_use_simplify (program : Tensor_ir.program) =
  let reachable = reachable_node_ids program in
  let nodes =
    List.filter
      (function
        | Tensor_ir.Elementwise1D { id; _ } | Reduce1D { id; _ } ->
            Int_set.mem id reachable)
      program.Tensor_ir.nodes
  in
  { program with nodes }

let input_use_counts program =
  let add_ref counts = function
    | Tensor_ir.ParamRef _ -> counts
    | Tensor_ir.NodeRef id ->
        let count = Option.value (Int_map.find_opt id counts) ~default:0 in
        Int_map.add id (count + 1) counts
  in
  let counts =
    List.fold_left
      (fun counts node ->
        match node with
        | Tensor_ir.Elementwise1D { inputs; _ } ->
            List.fold_left
              (fun acc (binding : Tensor_ir.input_binding) ->
                add_ref acc binding.source)
              counts inputs
        | Reduce1D { source; _ } ->
            List.fold_left add_ref counts
              (Reduction_analysis.reduction_sources source))
      Int_map.empty program.Tensor_ir.nodes
  in
  match program.result with
  | Tensor_ir.TensorResult value | ScalarResult value -> add_ref counts value

let fuse_program ?(allow_clone = false) ?(clone_complexity_limit = max_int)
    ?(optimizations = Optimizations.none) (program : Tensor_ir.program) =
  let use_counts = input_use_counts program in
  let transformed = Hashtbl.create 16 in
  let fused_ids = ref Int_set.empty in
  let next_name base existing =
    let rec loop index =
      let candidate = Printf.sprintf "%s__f%d" base index in
      if
        List.exists
          (fun (binding : Tensor_ir.input_binding) ->
            String.equal binding.name candidate)
          existing
      then loop (index + 1)
      else candidate
    in
    loop 0
  in
  let inline_binding existing_inputs body scalar_params
      (binding : Tensor_ir.input_binding) =
    match binding.source with
    | Tensor_ir.ParamRef _ ->
        (existing_inputs @ [ binding ], body, scalar_params)
    | NodeRef producer_id -> (
        match
          ( Int_map.find_opt producer_id use_counts,
            Hashtbl.find_opt transformed producer_id )
        with
        | Some use_count, Some (Tensor_ir.Elementwise1D producer) -> (
            let metrics = Reduction_analysis.body_metrics use_count producer.body in
            match
              Materialization_policy.allow_inline_producer optimizations metrics
                use_count
            with
            | `Disallow -> (existing_inputs @ [ binding ], body, scalar_params)
            | `Clone
              when
                not allow_clone
                || Reduction_analysis.scalar_expr_complexity producer.body
                   > clone_complexity_limit ->
                (existing_inputs @ [ binding ], body, scalar_params)
            | (`Fuse | `Clone) ->
                if use_count = 1 then
                  fused_ids := Int_set.add producer_id !fused_ids;
                let renamed_inputs, renamings =
                  List.fold_left
                    (fun (inputs, renamings)
                         (producer_input : Tensor_ir.input_binding) ->
                      let fresh =
                        next_name
                          (binding.name ^ "__" ^ producer_input.name)
                          (existing_inputs @ inputs)
                      in
                      ( inputs @ [ { producer_input with name = fresh } ],
                        renamings @ [ (producer_input.name, fresh) ] ))
                    ([], []) producer.inputs
                in
                let replacement =
                  Reduction_analysis.rename_scalar_expr_vars renamings
                    producer.body
                in
                ( existing_inputs @ renamed_inputs,
                  Reduction_analysis.substitute_scalar_expr binding.name
                    replacement body,
                  List.sort_uniq String.compare
                    (scalar_params @ producer.scalar_params) ))
        | _ -> (existing_inputs @ [ binding ], body, scalar_params))
  in
  let nodes =
    List.map
      (function
        | Tensor_ir.Elementwise1D ({ inputs; scalar_params; body; _ } as node)
          ->
            let inputs, body, scalar_params =
              List.fold_left
                (fun (acc_inputs, acc_body, acc_scalars) binding ->
                  inline_binding acc_inputs acc_body acc_scalars binding)
                ([], body, scalar_params) inputs
            in
            let node =
              Tensor_ir.Elementwise1D { node with inputs; scalar_params; body }
            in
            let id =
              match node with
              | Tensor_ir.Elementwise1D { id; _ } -> id
              | _ -> assert false
            in
            Hashtbl.replace transformed id node;
            node
        | Tensor_ir.Reduce1D _ as node ->
            let id =
              match node with
              | Tensor_ir.Reduce1D { id; _ } -> id
              | _ -> assert false
            in
            Hashtbl.replace transformed id node;
            node)
      program.Tensor_ir.nodes
    |> List.filter (function
           | Tensor_ir.Elementwise1D { id; _ } | Reduce1D { id; _ } ->
               not (Int_set.mem id !fused_ids))
  in
  { program with nodes }

let remap_value_ref redirects = function
  | Tensor_ir.ParamRef _ as value -> value
  | NodeRef id -> Option.value (Int_map.find_opt id redirects) ~default:(NodeRef id)

let remap_reduction_source redirects = function
  | Tensor_ir.PlainInput input -> PlainInput (remap_value_ref redirects input)
  | MappedInput ({ inputs; _ } as mapped) ->
      MappedInput
        {
          mapped with
          inputs =
            List.map
              (fun (binding : Tensor_ir.input_binding) ->
                { binding with source = remap_value_ref redirects binding.source })
              inputs;
        }

let reduce_map_fusion_program ?(allow_clone = false)
    ?(clone_complexity_limit = max_int) ?(optimizations = Optimizations.none)
    (program : Tensor_ir.program) =
  let use_counts = input_use_counts program in
  let by_id =
    List.fold_left
      (fun acc -> function
        | Tensor_ir.Elementwise1D { id; _ } as node -> Int_map.add id node acc
        | Reduce1D { id; _ } as node -> Int_map.add id node acc)
      Int_map.empty program.nodes
  in
  let fused_ids = ref Int_set.empty in
  let next_name base existing =
    let rec loop index =
      let candidate = Printf.sprintf "%s__r%d" base index in
      if
        List.exists
          (fun (binding : Tensor_ir.input_binding) ->
            String.equal binding.name candidate)
          existing
      then loop (index + 1)
      else candidate
    in
    loop 0
  in
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = PlainInput (NodeRef producer_id); _ } as node) ->
            begin
              match
                ( Int_map.find_opt producer_id use_counts,
                  Int_map.find_opt producer_id by_id )
              with
              | Some use_count, Some (Tensor_ir.Elementwise1D producer) -> (
                  let metrics = Reduction_analysis.body_metrics use_count producer.body in
                  match
                    Materialization_policy.allow_inline_into_reduction
                      optimizations metrics use_count 0
                  with
                  | `Disallow -> Reduce1D node
                  | `Clone
                    when
                      not allow_clone
                      || Reduction_analysis.scalar_expr_complexity producer.body
                         > clone_complexity_limit ->
                      Reduce1D node
                  | (`Fuse | `Clone) ->
                      if use_count = 1 then
                        fused_ids := Int_set.add producer_id !fused_ids;
                      let inputs, renamings =
                        List.fold_left
                          (fun (acc_inputs, acc_renamings)
                               (binding : Tensor_ir.input_binding) ->
                            let fresh =
                              next_name binding.name (acc_inputs @ producer.inputs)
                            in
                            ( acc_inputs @ [ { binding with name = fresh } ],
                              acc_renamings @ [ (binding.name, fresh) ] ))
                          ([], []) producer.inputs
                      in
                      let body =
                        Reduction_analysis.rename_scalar_expr_vars renamings
                          producer.body
                      in
                      Reduce1D
                        {
                          node with
                          source =
                            MappedInput
                              {
                                inputs;
                                scalar_params = producer.scalar_params;
                                body;
                              };
                        })
              | _ -> Reduce1D node
            end
        | node -> node)
      program.nodes
    |> List.filter (function
           | Tensor_ir.Elementwise1D { id; _ } -> not (Int_set.mem id !fused_ids)
           | Reduce1D _ -> true)
  in
  { program with nodes }

let reduction_body_normalize (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            let scalar_params =
              List.sort_uniq String.compare mapped.scalar_params
            in
            let inputs =
              List.sort (fun a b -> String.compare a.name b.name) mapped.inputs
            in
            let body =
              mapped.body
              |> Reduction_analysis.canonicalize_scalar_expr
              |> Reduction_analysis.simplify_scalar_expr
            in
            Reduce1D
              { node with source = MappedInput { inputs; scalar_params; body } }
        | node -> node)
      program.nodes
  in
  { program with nodes }

let reduction_reuse_hoist (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            let body =
              mapped.body
              |> Reduction_analysis.simplify_scalar_expr
              |> Reduction_analysis.canonicalize_scalar_expr
              |> Reduction_analysis.simplify_scalar_expr
            in
            Reduce1D
              { node with source = MappedInput { mapped with body } }
        | node -> node)
      program.nodes
  in
  { program with nodes }

let reduction_body_canonicalize (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            let body =
              mapped.body
              |> Reduction_analysis.canonicalize_reduction_body
                   mapped.scalar_params
              |> Reduction_analysis.simplify_scalar_expr
              |> Reduction_analysis.canonicalize_scalar_expr
            in
            Reduce1D { node with source = MappedInput { mapped with body } }
        | node -> node)
      program.nodes
  in
  { program with nodes }

let branchy_reduction_rescue (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            let body =
              mapped.body
              |> Reduction_analysis.branchy_weighted_rescue_body
                   mapped.scalar_params
              |> Reduction_analysis.simplify_scalar_expr
              |> Reduction_analysis.canonicalize_scalar_expr
            in
            Reduce1D { node with source = MappedInput { mapped with body } }
        | node -> node)
      program.nodes
  in
  { program with nodes }

let reduction_body_cse ?(optimizations = Optimizations.none)
    (program : Tensor_ir.program) =
  let nodes =
    List.map
      (function
        | Reduce1D ({ source = MappedInput mapped; _ } as node) ->
            let repeated =
              Reduction_analysis.scalar_expr_repeated_subexpressions mapped.body
            in
            let body =
              if
                repeated
                >= optimizations.config.reduction_body_cse_min_occurrences
              then
                mapped.body
                |> Reduction_analysis.reduction_body_cse_expr
                |> Reduction_analysis.simplify_scalar_expr
                |> Reduction_analysis.canonicalize_scalar_expr
              else mapped.body
            in
            Reduce1D { node with source = MappedInput { mapped with body } }
        | node -> node)
      program.nodes
  in
  { program with nodes }

let tensor_cse (program : Tensor_ir.program) =
  let key_of_value_ref = function
    | Tensor_ir.ParamRef name -> "p:" ^ name
    | NodeRef id -> Printf.sprintf "n:%d" id
  in
  let key_of_bindings bindings =
    bindings
    |> List.map (fun (binding : Tensor_ir.input_binding) ->
           binding.name ^ "=" ^ key_of_value_ref binding.source)
    |> String.concat "|"
  in
  let key_of_node = function
    | Tensor_ir.Elementwise1D { shape_symbol; inputs; scalar_params; body; _ } ->
        Printf.sprintf "e:%s:%s:%s:%s" shape_symbol (key_of_bindings inputs)
          (String.concat "," scalar_params)
          (Reduction_analysis.scalar_expr_rank body)
    | Reduce1D { shape_symbol; source; kind; _ } ->
        let kind_text =
          match kind with Tensor_ir.Sum -> "sum" | MaxReduce -> "max"
        in
        let source_text =
          match source with
          | PlainInput input -> "p:" ^ key_of_value_ref input
          | MappedInput { inputs; scalar_params; body } ->
              Printf.sprintf "m:%s:%s:%s" (key_of_bindings inputs)
                (String.concat "," scalar_params)
                (Reduction_analysis.scalar_expr_rank body)
        in
        Printf.sprintf "r:%s:%s:%s" shape_symbol kind_text source_text
  in
  let redirects = ref Int_map.empty in
  let seen = Hashtbl.create 32 in
  let keep_nodes =
    List.filter_map
      (fun node ->
        let node =
          match node with
          | Tensor_ir.Elementwise1D ({ inputs; _ } as item) ->
              Tensor_ir.Elementwise1D
                {
                  item with
                  inputs =
                    List.map
                      (fun (binding : Tensor_ir.input_binding) ->
                        {
                          binding with
                          source = remap_value_ref !redirects binding.source;
                        })
                      inputs;
                }
          | Reduce1D ({ source; _ } as item) ->
              Reduce1D
                { item with source = remap_reduction_source !redirects source }
        in
        let id =
          match node with
          | Tensor_ir.Elementwise1D { id; _ } | Reduce1D { id; _ } -> id
        in
        let key = key_of_node node in
        match Hashtbl.find_opt seen key with
        | Some existing ->
            redirects := Int_map.add id (NodeRef existing) !redirects;
            None
        | None ->
            Hashtbl.replace seen key id;
            Some node)
      program.nodes
  in
  let result =
    match program.result with
    | Tensor_ir.TensorResult value -> TensorResult (remap_value_ref !redirects value)
    | ScalarResult value -> ScalarResult (remap_value_ref !redirects value)
  in
  { program with nodes = keep_nodes; result }

let annotate_program ?(optimizations = Optimizations.none)
    ~(empty_metrics : Tensor_ir.body_metrics) (program : Tensor_ir.program) =
  let counts = input_use_counts program in
  let nodes =
    List.map
      (function
        | Tensor_ir.Elementwise1D ({ body; _ } as node) ->
            let uses = Option.value (Int_map.find_opt node.id counts) ~default:0 in
            let metrics = Reduction_analysis.body_metrics uses body in
            let handling_hint =
              Materialization_policy.materialization_hint optimizations metrics None
            in
            Elementwise1D { node with metrics; handling_hint }
        | Reduce1D ({ source; _ } as node) ->
            let uses = Option.value (Int_map.find_opt node.id counts) ~default:0 in
            let metrics =
              match source with
              | PlainInput _ -> { empty_metrics with estimated_uses = uses }
              | MappedInput { body; scalar_params; _ } ->
                  if
                    Optimizations.enabled optimizations
                      Optimizations.ReductionBodyCanonicalize
                  then
                    Reduction_analysis.adjusted_reduction_body_metrics
                      scalar_params uses body
                  else Reduction_analysis.body_metrics uses body
            in
            let reduction_shape =
              Reduction_analysis.classify_reduction_shape source
            in
            let handling_hint =
              Materialization_policy.materialization_hint optimizations metrics
                (Some reduction_shape)
            in
            Reduce1D { node with metrics; reduction_shape; handling_hint })
      program.nodes
  in
  { program with nodes }

let apply_optimizations ?(optimizations = Optimizations.none)
    ~(empty_metrics : Tensor_ir.body_metrics) program =
  let program =
    if Optimizations.enabled optimizations Optimizations.TensorCanonicalize then
      canonicalize_program program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionInputSimplify
    then simplify_program program
    else program
  in
  let program =
    if
      Optimizations.enabled optimizations Optimizations.ReductionBodyNormalize
      || Optimizations.enabled optimizations
           Optimizations.ReductionAccumulatorShape
    then reduction_body_normalize program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionReuseHoist then
      reduction_reuse_hoist program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.BranchyReductionRescue then
      branchy_reduction_rescue program
    else program
  in
  let program =
    if
      Optimizations.enabled optimizations Optimizations.ReductionBodyCanonicalize
    then reduction_body_canonicalize program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionBodyCse then
      reduction_body_cse ~optimizations program
    else program
  in
  let program =
    if
      Optimizations.enabled optimizations Optimizations.MapChainCollapse
      || Optimizations.enabled optimizations Optimizations.ElementwiseFusion
      || Optimizations.enabled optimizations Optimizations.SmallProducerClone
    then
      fuse_program
        ~allow_clone:
          (Optimizations.enabled optimizations Optimizations.SmallProducerClone)
        ~clone_complexity_limit:optimizations.config.small_clone_nodes
        ~optimizations program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReduceMapFusion then
      reduce_map_fusion_program
        ~allow_clone:
          (Optimizations.enabled optimizations Optimizations.SmallProducerClone)
        ~clone_complexity_limit:optimizations.config.small_clone_nodes
        ~optimizations program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ShapeUseSimplify then
      shape_use_simplify program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.TensorCse then
      tensor_cse program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.TensorCanonicalize then
      canonicalize_program program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionInputSimplify
    then simplify_program program
    else program
  in
  let program =
    if
      Optimizations.enabled optimizations Optimizations.ReductionBodyNormalize
      || Optimizations.enabled optimizations
           Optimizations.ReductionAccumulatorShape
    then reduction_body_normalize program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionReuseHoist then
      reduction_reuse_hoist program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.BranchyReductionRescue then
      branchy_reduction_rescue program
    else program
  in
  let program =
    if
      Optimizations.enabled optimizations Optimizations.ReductionBodyCanonicalize
    then reduction_body_canonicalize program
    else program
  in
  let program =
    if Optimizations.enabled optimizations Optimizations.ReductionBodyCse then
      reduction_body_cse ~optimizations program
    else program
  in
  annotate_program ~optimizations ~empty_metrics program
