module Kernel_plan = Triton_plan

type generated_module = { filename : string; source : string }

let py_float value =
  let text = Printf.sprintf "%.12g" value in
  if String.contains text '.' then text else text ^ ".0"

let rec render_scalar_expr = function
  | Tensor_ir.SVar name -> name
  | Tensor_ir.SConstF32 value -> py_float value
  | Tensor_ir.SConstBool value -> if value then "True" else "False"
  | Tensor_ir.SUnary (Neg, expr) ->
      Printf.sprintf "(-(%s))" (render_scalar_expr expr)
  | Tensor_ir.SUnary (Sqrt, expr) ->
      Printf.sprintf "tl.sqrt(%s)" (render_scalar_expr expr)
  | Tensor_ir.SUnary (Exp, expr) ->
      Printf.sprintf "tl.exp(%s)" (render_scalar_expr expr)
  | Tensor_ir.SUnary (Log, expr) ->
      Printf.sprintf "tl.log(%s)" (render_scalar_expr expr)
  | Tensor_ir.SBinary (Add, lhs, rhs) ->
      Printf.sprintf "((%s) + (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (Sub, lhs, rhs) ->
      Printf.sprintf "((%s) - (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (Mul, lhs, rhs) ->
      Printf.sprintf "((%s) * (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (Div, lhs, rhs) ->
      Printf.sprintf "((%s) / (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (Min, lhs, rhs) ->
      Printf.sprintf "tl.minimum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (Max, lhs, rhs) ->
      Printf.sprintf "tl.maximum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (CmpLt, lhs, rhs) ->
      Printf.sprintf "((%s) < (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (CmpLe, lhs, rhs) ->
      Printf.sprintf "((%s) <= (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (CmpGt, lhs, rhs) ->
      Printf.sprintf "((%s) > (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (CmpGe, lhs, rhs) ->
      Printf.sprintf "((%s) >= (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SBinary (CmpEq, lhs, rhs) ->
      Printf.sprintf "((%s) == (%s))" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SSelect (SBinary ((CmpLt | CmpLe), lhs, rhs), then_expr, else_expr)
    when then_expr = lhs && else_expr = rhs ->
      Printf.sprintf "tl.minimum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SSelect (SBinary ((CmpLt | CmpLe), lhs, rhs), then_expr, else_expr)
    when then_expr = rhs && else_expr = lhs ->
      Printf.sprintf "tl.maximum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SSelect (SBinary ((CmpGt | CmpGe), lhs, rhs), then_expr, else_expr)
    when then_expr = lhs && else_expr = rhs ->
      Printf.sprintf "tl.maximum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SSelect (SBinary ((CmpGt | CmpGe), lhs, rhs), then_expr, else_expr)
    when then_expr = rhs && else_expr = lhs ->
      Printf.sprintf "tl.minimum(%s, %s)" (render_scalar_expr lhs)
        (render_scalar_expr rhs)
  | Tensor_ir.SSelect (cond, then_expr, else_expr) ->
      Printf.sprintf "tl.where(%s, %s, %s)" (render_scalar_expr cond)
        (render_scalar_expr then_expr)
        (render_scalar_expr else_expr)

let find_node program node_id =
  match
    List.find_opt
      (function
        | Tensor_ir.Elementwise1D { id; _ } | Reduce1D { id; _ } -> id = node_id)
      program.Tensor_ir.nodes
  with
  | Some node -> node
  | None -> invalid_arg "missing tensor ir node for kernel plan step"

let join_lines lines = String.concat "\n" lines ^ "\n"

let indent_lines prefix lines = List.map (fun line -> prefix ^ line) lines

let render_triton_config (config : Autotune_config.config) =
  Printf.sprintf
    "triton.Config({\"BLOCK_SIZE\": %d}, num_warps=%d, num_stages=%d)"
    config.block_size config.num_warps config.num_stages

let render_triton_configs name family =
  join_lines
    [
      Printf.sprintf "%s = [" name;
      family.Autotune_config.configs
      |> List.map (fun config -> "    " ^ render_triton_config config ^ ",")
      |> String.concat "\n";
      "]";
      "";
    ]

let min_block_size family =
  family.Autotune_config.configs
  |> List.map (fun (config : Autotune_config.config) -> config.block_size)
  |> List.fold_left min max_int

let render_autotune_helpers (config : Autotune_config.t) =
  let bucket_lines =
    config.bucket_upper_bounds
    |> List.mapi (fun index bound ->
        Printf.sprintf "    if n <= %d:\n        return %d" bound index)
  in
  join_lines
    ([
       render_triton_configs "_LOOM_ELEMENTWISE_CONFIGS" config.elementwise;
       render_triton_configs "_LOOM_REDUCTION_CONFIGS" config.reduction;
       Printf.sprintf "_LOOM_REDUCTION_MIN_BLOCK_SIZE = %d"
         (min_block_size config.reduction);
       "";
       "def _loom_size_bucket(n):";
     ]
    @ bucket_lines
    @ [
        Printf.sprintf "    return %d" (List.length config.bucket_upper_bounds);
        "";
        "def _loom_selected_block_size(kernel, default, size_bucket=None):";
        "    cache = getattr(kernel, \"cache\", {})";
        "    if size_bucket is not None:";
        "        for key, config in cache.items():";
        "            if isinstance(key, tuple) and len(key) > 0 and key[0] == size_bucket:";
        "                kwargs = getattr(config, \"kwargs\", {})";
        "                return int(kwargs.get(\"BLOCK_SIZE\", default))";
        "    config = getattr(kernel, \"best_config\", None)";
        "    if config is None:";
        "        return default";
        "    kwargs = getattr(config, \"kwargs\", {})";
        "    return int(kwargs.get(\"BLOCK_SIZE\", default))";
        "";
      ])

let render_autotune_metadata (config : Autotune_config.t) =
  let source_path =
    match config.source_path with
    | Some path -> Printf.sprintf "%S" path
    | None -> "None"
  in
  let buckets =
    config.bucket_upper_bounds |> List.map string_of_int |> String.concat ", "
  in
  join_lines
    [
      "_LOOM_AUTOTUNE_METADATA = {";
      Printf.sprintf "    \"source_path\": %s," source_path;
      "    \"enabled\": True,";
      Printf.sprintf "    \"bucket_upper_bounds\": [%s]," buckets;
      "}";
      "";
    ]

let render_fixed_metadata =
  join_lines
    [
      "_LOOM_AUTOTUNE_METADATA = {";
      "    \"source_path\": None,";
      "    \"enabled\": False,";
      "    \"bucket_upper_bounds\": [],";
      "}";
      "";
    ]

let render_launch_bucket_list buckets =
  let items =
    buckets
    |> List.map (fun (bucket : Kernel_plan.launch_bucket) ->
        Printf.sprintf "(%d, %d, %d)" bucket.max_n bucket.block_size
          bucket.num_warps)
    |> String.concat ", "
  in
  "[" ^ items ^ "]"

let render_fixed_launch_helpers =
  join_lines
    [
      "def _loom_fixed_launch(n, default_block_size, default_num_warps, \
       buckets):";
      "    for max_n, block_size, num_warps in buckets:";
      "        if n <= max_n:";
      "            return int(block_size), int(num_warps)";
      "    return int(default_block_size), int(default_num_warps)";
      "";
    ]

let render_elementwise_kernel program (step : Kernel_plan.elementwise_step)
    autotune =
  let node =
    match find_node program step.node_id with
    | Tensor_ir.Elementwise1D { body; _ } -> body
    | _ -> invalid_arg "expected elementwise node"
  in
  let base_args = [ "out_ptr"; "n" ] in
  let args =
    base_args
    @ (match autotune with Some _ -> [ "size_bucket" ] | None -> [])
    @ List.map (fun (name, _) -> name ^ "_ptr") step.inputs
    @ step.scalar_params
  in
  let decorator_lines =
    match autotune with
    | None -> [ "@triton.jit" ]
    | Some _ ->
        [
          "@triton.autotune(configs=_LOOM_ELEMENTWISE_CONFIGS, \
           key=[\"size_bucket\"])";
          "@triton.jit";
        ]
  in
  let loads =
    List.map
      (fun (name, _) ->
        Printf.sprintf "    %s = tl.load(%s_ptr + offs, mask=mask, other=0.0)"
          name name)
      step.inputs
  in
  let body = render_scalar_expr node in
  join_lines
    (decorator_lines
    @ [
        Printf.sprintf "def %s(%s, BLOCK_SIZE: tl.constexpr):" step.kernel_name
          (String.concat ", " args);
        "    pid = tl.program_id(0)";
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)";
        "    mask = offs < n";
      ]
    @ loads
    @ [
        Printf.sprintf "    value = %s" body;
        "    tl.store(out_ptr + offs, value, mask=mask)";
        "";
      ])

let render_reduction_first_kernel (step : Kernel_plan.reduction_step) autotune =
  let identity =
    match step.reduce_kind with
    | Tensor_ir.Sum -> "0.0"
    | Tensor_ir.MaxReduce -> "-float(\"inf\")"
  in
  let combine =
    match step.reduce_kind with
    | Tensor_ir.Sum -> "tl.sum(values, axis=0)"
    | Tensor_ir.MaxReduce -> "tl.max(values, axis=0)"
  in
  let args =
    match step.source with
    | Kernel_plan.PlainInput _ ->
        [ "input_ptr"; "partial_ptr"; "n" ]
        @ (match autotune with Some _ -> [ "size_bucket" ] | None -> [])
    | MappedInput { inputs; scalar_params; _ } ->
        [ "partial_ptr"; "n" ]
        @ (match autotune with Some _ -> [ "size_bucket" ] | None -> [])
        @ List.map (fun (name, _) -> name ^ "_ptr") inputs
        @ scalar_params
  in
  let decorator_lines =
    match autotune with
    | None -> [ "@triton.jit" ]
    | Some _ ->
        [
          "@triton.autotune(configs=_LOOM_REDUCTION_CONFIGS, \
           key=[\"size_bucket\"])";
          "@triton.jit";
        ]
  in
  let standard_lines =
    [
      Printf.sprintf "def %s(%s, BLOCK_SIZE: tl.constexpr):" step.kernel_name
        (String.concat ", " args);
      "    pid = tl.program_id(0)";
      "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)";
      "    mask = offs < n";
    ]
    @ (match step.source with
      | Kernel_plan.PlainInput _ ->
          [
            Printf.sprintf
              "    values = tl.load(input_ptr + offs, mask=mask, other=%s)"
              identity;
          ]
      | MappedInput { inputs; body; _ } ->
          List.map
            (fun (name, _) ->
              Printf.sprintf
                "    %s = tl.load(%s_ptr + offs, mask=mask, other=0.0)" name
                name)
            inputs
          @ [ Printf.sprintf "    values = %s" (render_scalar_expr body) ])
    @ [
        Printf.sprintf "    partial = %s" combine;
        "    tl.store(partial_ptr + pid, partial)";
        "";
      ]
  in
  let strided_lines =
    [
      Printf.sprintf "def %s(%s, BLOCK_SIZE: tl.constexpr):" step.kernel_name
        (String.concat ", " args);
      "    pid = tl.program_id(0)";
      "    program_count = tl.num_programs(0)";
      "    offset = pid * BLOCK_SIZE";
      "    stride = program_count * BLOCK_SIZE";
      Printf.sprintf "    acc = %s" identity;
      "    while offset < n:";
      "        offs = offset + tl.arange(0, BLOCK_SIZE)";
      "        mask = offs < n";
    ]
    @ (match step.source with
      | Kernel_plan.PlainInput _ ->
          [
            Printf.sprintf
              "        values = tl.load(input_ptr + offs, mask=mask, other=%s)"
              identity;
          ]
      | MappedInput { inputs; body; _ } ->
          List.map
            (fun (name, _) ->
              Printf.sprintf
                "        %s = tl.load(%s_ptr + offs, mask=mask, other=0.0)" name
                name)
            inputs
          @ [ Printf.sprintf "        values = %s" (render_scalar_expr body) ])
    @ [
        Printf.sprintf "        partial = %s" combine;
        (match step.reduce_kind with
        | Tensor_ir.Sum -> "        acc += partial"
        | Tensor_ir.MaxReduce -> "        acc = tl.maximum(acc, partial)");
        "        offset += stride";
        "    tl.store(partial_ptr + pid, acc)";
        "";
      ]
  in
  join_lines
    (decorator_lines
    @
    if
      step.reduction_strategy = Kernel_plan.SmallDirectReduction
      || step.reduction_strategy = Kernel_plan.SmallPartialReduction
    then strided_lines
    else standard_lines)

let render_reduction_combine_kernel (step : Kernel_plan.reduction_step) autotune
    =
  let identity =
    match step.reduce_kind with
    | Tensor_ir.Sum -> "0.0"
    | Tensor_ir.MaxReduce -> "-float(\"inf\")"
  in
  let combine =
    match step.reduce_kind with
    | Tensor_ir.Sum -> "tl.sum(values, axis=0)"
    | Tensor_ir.MaxReduce -> "tl.max(values, axis=0)"
  in
  let args =
    [ "input_ptr"; "partial_ptr"; "n" ]
    @ match autotune with Some _ -> [ "size_bucket" ] | None -> []
  in
  let decorator_lines =
    match autotune with
    | None -> [ "@triton.jit" ]
    | Some _ ->
        [
          "@triton.autotune(configs=_LOOM_REDUCTION_CONFIGS, \
           key=[\"size_bucket\"])";
          "@triton.jit";
        ]
  in
  join_lines
    (decorator_lines
    @ [
        Printf.sprintf "def %s(%s, BLOCK_SIZE: tl.constexpr):"
          step.combine_kernel_name (String.concat ", " args);
        "    pid = tl.program_id(0)";
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)";
        "    mask = offs < n";
        Printf.sprintf
          "    values = tl.load(input_ptr + offs, mask=mask, other=%s)" identity;
        Printf.sprintf "    partial = %s" combine;
        "    tl.store(partial_ptr + pid, partial)";
        "";
      ])

let render_checks tensor_params =
  let body =
    [
      "def _check_tensor(name, value):";
      "    if not isinstance(value, torch.Tensor):";
      "        raise TypeError(f\"{name} must be a torch.Tensor\")";
      "    if not value.is_cuda:";
      "        raise ValueError(f\"{name} must be a CUDA tensor\")";
      "    if value.dtype != torch.float32:";
      "        raise ValueError(f\"{name} must have dtype torch.float32\")";
      "    if value.ndim != 1:";
      "        raise ValueError(f\"{name} must be rank-1\")";
      "    if not value.is_contiguous():";
      "        raise ValueError(f\"{name} must be contiguous\")";
      "";
    ]
  in
  let checks =
    tensor_params
    |> List.map (fun name ->
        Printf.sprintf "    _check_tensor(%S, %s)" name name)
  in
  let shape_checks =
    match tensor_params with
    | [] | [ _ ] -> []
    | first :: rest ->
        [ Printf.sprintf "    n = %s.numel()" first ]
        @ List.map
            (fun name ->
              Printf.sprintf
                "    if %s.numel() != n:\n\
                \        raise ValueError(\"all tensor inputs must have the \
                 same shape\")"
                name)
            rest
  in
  (body, checks @ shape_checks)

let tensor_param_names params =
  params
  |> List.filter_map (function
    | Tensor_ir.Tensor1F32 (name, _) -> Some name
    | _ -> None)

let scalar_param_names params =
  params
  |> List.filter_map (function
    | Tensor_ir.ScalarF32 name -> Some name
    | _ -> None)

let triton_uncapped_small_partial_family = function
  | "mapped" | "weighted" | "mapped-reuse" | "norm-square"
  | "affine-norm-square" | "dot-product" | "weighted-product"
  | "delta-square" ->
      true
  | _ -> false

let triton_fixed_reduction_launch_buckets step =
  if triton_uncapped_small_partial_family step.Kernel_plan.reduction_family then
    step.launch_buckets
    |> List.map (fun (bucket : Kernel_plan.launch_bucket) ->
           if bucket.block_size < step.block_size then
             { bucket with block_size = step.block_size }
           else bucket)
  else step.launch_buckets

let reduction_device_expr (step : Kernel_plan.reduction_step) =
  match step.source with
  | Kernel_plan.PlainInput input -> input ^ ".device"
  | Kernel_plan.MappedInput { inputs; _ } -> (
      match inputs with
      | (_, source) :: _ -> source ^ ".device"
      | [] -> "torch.device('cuda')")

let render_step_call step autotune =
  match step with
  | Kernel_plan.Elementwise step ->
      let grid_lines =
        match autotune with
        | None ->
            if step.launch_buckets = [] then
              [
                Printf.sprintf "    block_size, num_warps = %d, %d"
                  step.block_size step.num_warps;
                "    grid = (triton.cdiv(n, block_size),)";
              ]
            else
              [
                Printf.sprintf
                  "    block_size, num_warps = _loom_fixed_launch(n, %d, %d, \
                   %s)"
                  step.block_size step.num_warps
                  (render_launch_bucket_list step.launch_buckets);
                "    grid = (triton.cdiv(n, block_size),)";
              ]
        | Some _ ->
            [
              "    grid = lambda META: (triton.cdiv(n, META[\"BLOCK_SIZE\"]),)";
            ]
      in
      let call_args =
        [ step.output; "n" ]
        @ (match autotune with
          | Some _ -> [ "_loom_size_bucket(n)" ]
          | None -> [])
        @ List.map snd step.inputs @ step.scalar_params
      in
      let call_line =
        match autotune with
        | None ->
            Printf.sprintf
              "    %s[grid](%s, BLOCK_SIZE=block_size, num_warps=num_warps)"
              step.kernel_name
              (String.concat ", " call_args)
        | Some _ ->
            Printf.sprintf "    %s[grid](%s)" step.kernel_name
              (String.concat ", " call_args)
      in
      grid_lines @ [ call_line ]
  | Kernel_plan.Reduction step ->
      let device_expr = reduction_device_expr step in
      let plain_input_name =
        match step.source with
        | Kernel_plan.PlainInput input -> Some input
        | Kernel_plan.MappedInput _ -> None
      in
      let first_pass_loop_args =
        match step.source with
        | Kernel_plan.PlainInput _ -> (
            [ "current"; "partial"; "current_n" ]
            @
            match autotune with
            | Some _ -> [ "_loom_size_bucket(current_n)" ]
            | None -> [])
        | Kernel_plan.MappedInput { inputs; scalar_params; _ } ->
            [ "partial"; "n" ]
            @ (match autotune with
              | Some _ -> [ "_loom_size_bucket(n)" ]
              | None -> [])
            @ List.map snd inputs @ scalar_params
      in
      let first_pass_direct_args =
        match step.source with
        | Kernel_plan.PlainInput _ -> (
            [ "current"; "current"; "current_n" ]
            @
            match autotune with
            | Some _ -> [ "_loom_size_bucket(current_n)" ]
            | None -> [])
        | Kernel_plan.MappedInput { inputs; scalar_params; _ } ->
            [ "current"; "n" ]
            @ (match autotune with
              | Some _ -> [ "_loom_size_bucket(n)" ]
              | None -> [])
            @ List.map snd inputs @ scalar_params
      in
      let fixed_launch_buckets = triton_fixed_reduction_launch_buckets step in
      let fixed_bucket_line =
        if fixed_launch_buckets = [] then
          Printf.sprintf "        block_size, num_warps = %d, %d"
            step.block_size step.num_warps
        else
          Printf.sprintf
            "        block_size, num_warps = _loom_fixed_launch(current_n, %d, \
             %d, %s)"
            step.block_size step.num_warps
            (render_launch_bucket_list fixed_launch_buckets)
      in
      let first_pass_loop_call =
        match autotune with
        | None ->
            Printf.sprintf
              "        %s[grid](%s, BLOCK_SIZE=block_size, num_warps=num_warps)"
              step.kernel_name
              (String.concat ", " first_pass_loop_args)
        | Some _ ->
            Printf.sprintf "        %s[grid](%s)" step.kernel_name
              (String.concat ", " first_pass_loop_args)
      in
      let first_pass_direct_call =
        match autotune with
        | None ->
            Printf.sprintf
              "        %s[grid](%s, BLOCK_SIZE=block_size, num_warps=num_warps)"
              step.kernel_name
              (String.concat ", " first_pass_direct_args)
        | Some _ ->
            Printf.sprintf "        %s[grid](%s)" step.kernel_name
              (String.concat ", " first_pass_direct_args)
      in
      let combine_call =
        match autotune with
        | None ->
            Printf.sprintf
              "        %s[grid](current, partial, current_n, \
               BLOCK_SIZE=block_size, num_warps=num_warps)"
              step.combine_kernel_name
        | Some _ ->
            Printf.sprintf
              "        %s[grid](current, partial, current_n, \
               _loom_size_bucket(current_n))"
              step.combine_kernel_name
      in
      let active_update kernel_name bucket_expr =
        match autotune with
        | None ->
            [
              "        current = partial"; "        current_n = current.numel()";
            ]
        | Some _ ->
            [
              Printf.sprintf
                "        active = triton.cdiv(current_n, \
                 _loom_selected_block_size(%s, \
                 _LOOM_REDUCTION_MIN_BLOCK_SIZE, %s))"
                kernel_name bucket_expr;
              "        current = partial[:active]";
              "        current_n = current.numel()";
            ]
      in
      let direct_lines =
        match step.single_block_threshold with
        | None -> []
        | Some threshold ->
            let grid_line =
              match autotune with
              | None -> "        grid = (1,)"
              | Some _ -> "        grid = lambda META: (1,)"
            in
            [
              Printf.sprintf "    if current_n <= %d:" threshold;
              Printf.sprintf
                "        current = torch.empty((1,), device=%s, \
                 dtype=torch.float32)"
                device_expr;
            ]
            @ (match autotune with
              | None -> [ fixed_bucket_line ]
              | Some _ -> [])
            @ [
                grid_line;
                first_pass_direct_call;
                Printf.sprintf "        %s = current" step.output;
              ]
            @ [ Printf.sprintf "        return %s" step.output ]
      in
      let small_lines =
        match
          ( step.small_reduction_threshold,
            step.small_program_count,
            step.reduction_strategy )
        with
        | Some threshold, _, Kernel_plan.SmallDirectReduction ->
            let grid_line =
              match autotune with
              | None -> "        grid = (1,)"
              | Some _ -> "        grid = lambda META: (1,)"
            in
            [
              Printf.sprintf "    if current_n <= %d:" threshold;
              Printf.sprintf
                "        current = torch.empty((1,), device=%s, \
                 dtype=torch.float32)"
                device_expr;
            ]
            @ (match autotune with
              | None -> [ fixed_bucket_line ]
              | Some _ -> [])
            @ [
                grid_line;
                first_pass_direct_call;
                Printf.sprintf "        %s = current" step.output;
                Printf.sprintf "        return %s" step.output;
              ]
        | Some threshold, Some program_count, Kernel_plan.SmallPartialReduction
          ->
            let uncapped_small_partial =
              triton_uncapped_small_partial_family step.reduction_family
              && (step.complexity_bucket = Kernel_plan.Tiny
                 || step.complexity_bucket = Kernel_plan.Small)
            in
            let fixed_lines, active_line, grid_line, partial_line =
              match autotune with
              | None ->
                  ( [ fixed_bucket_line ],
                    (if uncapped_small_partial then
                       "        active = triton.cdiv(current_n, block_size)"
                     else
                       Printf.sprintf
                         "        active = min(%d, triton.cdiv(current_n, \
                          block_size))"
                         program_count),
                    "        grid = (active,)",
                    Printf.sprintf
                      "        partial = torch.empty((active,), device=%s, \
                       dtype=torch.float32)"
                      device_expr )
              | Some _ ->
                  let selected_block_size =
                    Printf.sprintf
                      "_loom_selected_block_size(%s, \
                       _LOOM_REDUCTION_MIN_BLOCK_SIZE, \
                       _loom_size_bucket(current_n))"
                      step.kernel_name
                  in
                  ( [],
                    (if uncapped_small_partial then
                       Printf.sprintf
                         "        active = triton.cdiv(current_n, %s)"
                         selected_block_size
                     else
                       Printf.sprintf
                         "        active = min(%d, triton.cdiv(current_n, %s))"
                         program_count selected_block_size),
                    (if uncapped_small_partial then
                       "        grid = lambda META: (triton.cdiv(current_n, \
                        META[\"BLOCK_SIZE\"]),)"
                     else
                       Printf.sprintf
                         "        grid = lambda META: (min(%d, \
                          triton.cdiv(current_n, META[\"BLOCK_SIZE\"])),)"
                         program_count),
                    Printf.sprintf
                      "        partial = torch.zeros((active,), device=%s, \
                       dtype=torch.float32)"
                      device_expr )
            in
            let nested_lines =
              (match autotune with None -> [ fixed_bucket_line ] | Some _ -> [])
              @ [
                  (match autotune with
                  | None ->
                      "        grid = (triton.cdiv(current_n, block_size),)"
                  | Some _ ->
                      "        grid = lambda META: (triton.cdiv(current_n, \
                       META[\"BLOCK_SIZE\"]),)");
                  (match autotune with
                  | None ->
                      "        partial = torch.empty((triton.cdiv(current_n, \
                       block_size),), device=current.device, dtype=torch.float32)"
                  | Some _ ->
                      "        partial = torch.zeros((triton.cdiv(current_n, \
                       _LOOM_REDUCTION_MIN_BLOCK_SIZE),), device=current.device, \
                       dtype=torch.float32)");
                  combine_call;
                ]
              @ active_update step.combine_kernel_name "_loom_size_bucket(current_n)"
            in
            [
              Printf.sprintf "    if current_n <= %d:" threshold;
            ]
            @ fixed_lines
            @ [
                active_line;
                partial_line;
                grid_line;
                first_pass_loop_call;
              ]
            @ active_update step.kernel_name "_loom_size_bucket(current_n)"
            @ [
                "        while current_n > 1:";
              ]
            @ indent_lines "    " nested_lines
            @ [
                Printf.sprintf "        %s = current" step.output;
                Printf.sprintf "        return %s" step.output;
              ]
        | _ -> []
      in
      let first_pass_lines =
        [ "    while True:" ]
        @ (match autotune with None -> [ fixed_bucket_line ] | Some _ -> [])
        @ [
            (match autotune with
            | None -> "        grid = (triton.cdiv(current_n, block_size),)"
            | Some _ ->
                "        grid = lambda META: (triton.cdiv(current_n, \
                 META[\"BLOCK_SIZE\"]),)");
            (match autotune with
            | None ->
                "        partial = torch.empty((triton.cdiv(current_n, \
                 block_size),), device=" ^ device_expr
                ^ ", dtype=torch.float32)"
            | Some _ ->
                "        partial = torch.zeros((triton.cdiv(current_n, \
                 _LOOM_REDUCTION_MIN_BLOCK_SIZE),), device=" ^ device_expr
                ^ ", dtype=torch.float32)");
            first_pass_loop_call;
          ]
        @ active_update step.kernel_name "_loom_size_bucket(current_n)"
        @ [ "        break" ]
      in
      let setup_lines =
        match plain_input_name with
        | Some input -> [ Printf.sprintf "    current = %s" input ]
        | None -> []
      in
      setup_lines @ [ "    current_n = n" ] @ direct_lines @ small_lines
      @ first_pass_lines
      @ [ "    while current_n > 1:" ]
      @ (match autotune with None -> [ fixed_bucket_line ] | Some _ -> [])
      @ [
          (match autotune with
          | None -> "        grid = (triton.cdiv(current_n, block_size),)"
          | Some _ ->
              "        grid = lambda META: (triton.cdiv(current_n, \
               META[\"BLOCK_SIZE\"]),)");
          (match autotune with
          | None ->
              "        partial = torch.empty((triton.cdiv(current_n, \
               block_size),), device=current.device, dtype=torch.float32)"
          | Some _ ->
              "        partial = torch.zeros((triton.cdiv(current_n, \
               _LOOM_REDUCTION_MIN_BLOCK_SIZE),), device=current.device, \
               dtype=torch.float32)");
          combine_call;
        ]
      @ active_update step.combine_kernel_name "_loom_size_bucket(current_n)"
      @ [ Printf.sprintf "    %s = current" step.output ]

let render_autotune_state program plan autotune =
  let kernels =
    plan.Kernel_plan.steps
    |> List.concat_map (function
      | Kernel_plan.Elementwise step -> [ (step.kernel_name, "elementwise") ]
      | Kernel_plan.Reduction step ->
          [
            (step.kernel_name, "reduction-first");
            (step.combine_kernel_name, "reduction-combine");
          ])
  in
  let kernel_entries =
    kernels
    |> List.map (fun (name, kind) ->
        Printf.sprintf
          "        %S: {\"kind\": %S, \"cache\": _loom_kernel_cache(%s)}," name
          kind name)
  in
  let helpers =
    [
      "def _loom_config_dict(config):";
      "    if config is None:";
      "        return None";
      "    return {";
      "        \"meta\": dict(getattr(config, \"kwargs\", {})),";
      "        \"num_warps\": getattr(config, \"num_warps\", None),";
      "        \"num_stages\": getattr(config, \"num_stages\", None),";
      "    }";
      "";
      "def _loom_kernel_cache(kernel):";
      "    cache = {}";
      "    for key, config in getattr(kernel, \"cache\", {}).items():";
      "        cache[str(key)] = _loom_config_dict(config)";
      "    return {";
      "        \"best_config\": _loom_config_dict(getattr(kernel, \
       \"best_config\", None)),";
      "        \"cache\": cache,";
      "    }";
      "";
    ]
  in
  let body =
    match autotune with
    | None ->
        [
          "def __loom_autotune_state__():";
          Printf.sprintf
            "    return {\"enabled\": False, \"kernels\": {}, \"entry\": %S}"
            program.Tensor_ir.entry_name;
        ]
    | Some _ ->
        [
          "def __loom_autotune_state__():";
          "    return {";
          Printf.sprintf "        \"enabled\": True,";
          Printf.sprintf "        \"entry\": %S," program.entry_name;
          "        \"kernels\": {";
        ]
        @ kernel_entries @ [ "        },"; "    }" ]
  in
  join_lines (helpers @ body @ [ "" ])

let render_wrapper program plan autotune =
  let tensor_params = tensor_param_names program.Tensor_ir.params in
  let scalar_params = scalar_param_names program.params in
  let all_params = scalar_params @ tensor_params in
  let check_fn, checks = render_checks tensor_params in
  let prologue =
    check_fn
    @ [
        Printf.sprintf "def %s(%s):" program.entry_name
          (String.concat ", " all_params);
      ]
    @ List.map
        (fun name -> Printf.sprintf "    %s = float(%s)" name name)
        scalar_params
    @ checks
    @
    match tensor_params with
    | [] -> [ "    n = 1" ]
    | first :: _ -> [ Printf.sprintf "    n = %s.numel()" first ]
  in
  let body =
    List.concat_map
      (function
        | Kernel_plan.Elementwise step ->
            let first_input =
              match step.inputs with
              | (_, source) :: _ -> source
              | [] -> failwith "elementwise step without input"
            in
            [
              Printf.sprintf "    %s = torch.empty_like(%s)" step.output
                first_input;
            ]
            @ render_step_call (Kernel_plan.Elementwise step) autotune
        | Kernel_plan.Reduction step ->
            render_step_call (Kernel_plan.Reduction step) autotune)
      plan.Kernel_plan.steps
  in
  let epilogue = [ Printf.sprintf "    return %s" plan.result_name; "" ] in
  join_lines (prologue @ body @ epilogue)

let generate ~autotune ~program ~plan =
  let prelude =
    join_lines
      [ "import torch"; "import triton"; "import triton.language as tl"; "" ]
  in
  let autotune_source =
    match autotune with
    | None -> render_fixed_launch_helpers ^ render_fixed_metadata
    | Some config ->
        render_fixed_launch_helpers
        ^ render_autotune_helpers config
        ^ render_autotune_metadata config
  in
  let kernels =
    List.map
      (function
        | Kernel_plan.Elementwise step ->
            render_elementwise_kernel program step autotune
        | Kernel_plan.Reduction step ->
            render_reduction_first_kernel step autotune
            ^ render_reduction_combine_kernel step autotune)
      plan.Kernel_plan.steps
    |> String.concat "\n"
  in
  let wrapper = render_wrapper program plan autotune in
  let autotune_state = render_autotune_state program plan autotune in
  {
    filename = program.entry_name ^ "_triton.py";
    source = prelude ^ autotune_source ^ kernels ^ wrapper ^ autotune_state;
  }
