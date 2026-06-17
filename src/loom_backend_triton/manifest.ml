module Triton_backend_plan = Triton_plan

let param_json = function
  | Tensor_ir.ScalarF32 name ->
      `Assoc [ ("name", `String name); ("kind", `String "scalar-f32") ]
  | Tensor_ir.Tensor1F32 (name, shape_symbol) ->
      `Assoc
        [ ("name", `String name)
        ; ("kind", `String "tensor1-f32")
        ; ("shape_symbol", `String shape_symbol) ]

let result_kind = function
  | Tensor_ir.TensorResult _ -> "tensor"
  | Tensor_ir.ScalarResult _ -> "scalar"

let render_front_ir_manifest ~front_entry ~lambda_entry ~program ~kernel_plan
    ~backend_plan ~optimizations =
  `Assoc
    [ ("front_ir", Front_ir.entry_to_yojson front_entry)
    ; ("loom_lambda", LoomLambda.entry_to_yojson lambda_entry)
    ; ("tensor_ir", Tensor_ir.program_to_yojson program)
    ; ("kernel_plan", Kernel_plan.to_yojson kernel_plan)
    ; ("triton_plan", Triton_backend_plan.to_yojson backend_plan)
    ; ("optimizations", Optimizations.to_yojson optimizations) ]
  |> Yojson.Safe.pretty_to_string

let render_manifest ~(program : Tensor_ir.program) ~(plan : Triton_plan.t)
    ~autotune ~generated_files ~optimizations =
  let autotune_json =
    match autotune with
    | Some config -> Autotune_config.to_yojson config
    | None -> `Null
  in
  `Assoc
    [ ("entry_name", `String program.Tensor_ir.entry_name)
    ; ("target_backend", `String "triton")
    ; ("params", `List (List.map param_json program.params))
    ; ("result_kind", `String (result_kind program.result))
    ; ("generated_files", `List (List.map (fun path -> `String path) generated_files))
    ; ("kernel_count", `Int (List.length plan.steps))
    ; ("temporary_count", `Int plan.temporary_count)
    ; ("optimizations", Optimizations.to_yojson optimizations)
    ; ("autotune", autotune_json) ]
  |> Yojson.Safe.pretty_to_string

let rec collect_tensor_ops acc = function
  | LoomLambda.TensorPrim (kind, args) ->
      List.fold_left collect_tensor_ops (LoomLambda.tensor_prim_to_string kind :: acc) args
  | LoomLambda.Let (_, value, body) -> collect_tensor_ops (collect_tensor_ops acc value) body
  | LoomLambda.If (cond, then_expr, else_expr) ->
      collect_tensor_ops (collect_tensor_ops (collect_tensor_ops acc cond) then_expr) else_expr
  | LoomLambda.Lambda (_, body) -> collect_tensor_ops acc body
  | LoomLambda.Apply (fn, args) ->
      List.fold_left collect_tensor_ops (collect_tensor_ops acc fn) args
  | LoomLambda.Prim (_, args) -> List.fold_left collect_tensor_ops acc args
  | LoomLambda.Var _ | FloatConst _ | BoolConst _ -> acc

let render_report ~front_entry ~lambda_entry ~(program : Tensor_ir.program)
    ~(plan : Triton_plan.t) ~autotune ~optimizations =
  let recognized_ops =
    collect_tensor_ops [] lambda_entry.LoomLambda.body
    |> List.rev |> List.sort_uniq String.compare
  in
  let autotune_lines =
    match autotune with
    | None ->
        [ "- Triton mode: fixed"
        ; "- launch config: BLOCK_SIZE=256, num_warps=4" ]
    | Some config ->
        [ "- Triton mode: autotuned"
        ; Printf.sprintf "- autotune source: `%s`"
            (match config.Autotune_config.source_path with
            | Some path -> path
            | None -> "<inline>")
        ; Printf.sprintf "- autotune buckets: %s"
            (config.bucket_upper_bounds
            |> List.map string_of_int
            |> String.concat ", ") ]
  in
  let enabled name = List.mem name (Optimizations.to_string_list optimizations) in
  let pipeline_notes =
    [ Printf.sprintf "- elementwise fusion: %s"
        (if enabled "elementwise-fusion" then "enabled" else "disabled")
    ; Printf.sprintf "- reduction precombine: %s"
        (if enabled "reduction-precombine" then "enabled" else "disabled")
    ; Printf.sprintf "- launch buckets: %s"
        (if enabled "launch-bucket-specialize" then "enabled" else "disabled") ]
  in
  String.concat "\n"
    ([ "# Loom Report"
    ; ""
    ; Printf.sprintf "- entrypoint: `%s`" program.Tensor_ir.entry_name
    ; "- pipeline: `FrontIR -> LoomLambda -> TensorIR -> KernelPlan traits -> TritonPlan -> Triton`"
    ; Printf.sprintf "- signature: `%s(%s)` -> `%s`" lambda_entry.name
        (String.concat ", "
           (List.map
              (fun (param : LoomLambda.param) ->
                Printf.sprintf "%s:%s" param.LoomLambda.name
                  (Loom_types.stage_type_to_string param.ty))
              lambda_entry.params))
        (Loom_types.stage_type_to_string lambda_entry.return_type)
    ; Printf.sprintf "- FrontIR body kind: `%s`"
        (match front_entry.Front_ir.body with
        | Front_ir.Let _ -> "let"
        | Front_ir.If _ -> "if"
        | Front_ir.Lambda _ -> "lambda"
        | Front_ir.Apply _ -> "apply"
        | Front_ir.TensorPrim _ -> "tensor-prim"
        | Front_ir.Prim _ -> "prim"
        | Front_ir.Tuple _ -> "tuple"
        | Front_ir.Var _ -> "var"
        | Front_ir.FloatConst _ -> "float"
        | Front_ir.BoolConst _ -> "bool"
        | Front_ir.UnitConst -> "unit")
    ; Printf.sprintf "- recognized tensor ops: %s"
        (if recognized_ops = [] then "none" else String.concat ", " recognized_ops)
    ; Printf.sprintf "- generated kernels: %d" (List.length plan.steps)
    ; Printf.sprintf "- temporary allocations: %d" plan.temporary_count
    ; Printf.sprintf "- enabled optimizations: %s"
        (match Optimizations.to_string_list optimizations with
        | [] -> "none"
        | names -> String.concat ", " names)
    ]
    @ autotune_lines
    @ pipeline_notes
    @ [ "" ])
