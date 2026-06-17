module Shared_kernel_plan = Kernel_plan
module Kernel_plan = Cuda_plan

type package_kind = Shared | Static

type cuda_platform =
  | GenericPlatform
  | CurrentPlatform

type frontend_kind =
  | OcamlFrontend
  | PythonFrontend
  | CppFrontend
  | AutoFrontend

type entry_bundle = {
  module_name : string;
  source_file : string;
  entry_name : string;
  symbol_name : string;
  workspace_symbol : string;
  lowered_sexp : string option;
  loom_entry_json : string;
  program : Tensor_ir.program;
  tensor_ir_json : string;
  plan : Kernel_plan.t;
  kernel_plan_json : string;
  cuda_plan_json : string;
  backend_analysis_json : string;
  optimizations : Optimizations.t;
  cuda_platform : cuda_platform;
}

type compile_result = {
  source_file : string;
  entry_name : string;
  module_name : string;
  symbol_name : string;
  workspace_symbol : string;
  artifact_path : string;
  header_path : string;
  source_path : string;
  manifest_path : string;
  report_path : string;
  generated_files : string list;
}

type packaged_result = {
  project_root : string;
  package_name : string;
  artifact_path : string;
  header_path : string;
  source_path : string;
  manifest_path : string;
  report_path : string;
}

let package_kind_to_string = function Shared -> "shared" | Static -> "static"
let cuda_platform_to_string = function
  | GenericPlatform -> "generic"
  | CurrentPlatform -> "current"

let frontend_kind_to_string = function
  | OcamlFrontend -> "ocaml"
  | PythonFrontend -> "python"
  | CppFrontend -> "cpp"
  | AutoFrontend -> "auto"

let parse_cuda_platform value =
  match String.lowercase_ascii value with
  | "generic" | "portable" -> GenericPlatform
  | "current" | "native" -> CurrentPlatform
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf
           "unknown CUDA platform %s (expected generic or current)" value)

let lowercase = String.lowercase_ascii
let quote = Filename.quote

let is_directory path =
  try (Unix.stat path).Unix.st_kind = Unix.S_DIR
  with Unix.Unix_error _ -> false

let ensure_dir path =
  let rec loop current =
    if current = "" || Sys.file_exists current then ()
    else (
      loop (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  loop path

let write_file path contents =
  ensure_dir (Filename.dirname path);
  let oc = open_out path in
  output_string oc contents;
  close_out oc

let run_command command =
  let status = Sys.command command in
  if status <> 0 then
    Diagnostic.raise_error
      (Printf.sprintf "command failed with exit status %d: %s" status command)

let command_available name =
  Sys.command (Printf.sprintf "command -v %s >/dev/null 2>&1" (quote name)) = 0

let rec find_project_root path =
  let absolute =
    if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
    else path
  in
  if not (Sys.file_exists absolute) then
    Diagnostic.raise_error (Printf.sprintf "path %S does not exist" path);
  let dir =
    if is_directory absolute then absolute else Filename.dirname absolute
  in
  let marker = Filename.concat dir "dune-project" in
  if Sys.file_exists marker then dir
  else
    let parent = Filename.dirname dir in
    if String.equal parent dir then
      Diagnostic.raise_error
        (Printf.sprintf "could not find dune-project above %s" path)
    else find_project_root parent

let rec find_marker_root marker path =
  let absolute =
    if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
    else path
  in
  if not (Sys.file_exists absolute) then
    Diagnostic.raise_error (Printf.sprintf "path %S does not exist" path);
  let dir =
    if is_directory absolute then absolute else Filename.dirname absolute
  in
  let marker_path = Filename.concat dir marker in
  if Sys.file_exists marker_path then dir
  else
    let parent = Filename.dirname dir in
    if String.equal parent dir then
      Diagnostic.raise_error
        (Printf.sprintf "could not find %s above %s" marker path)
    else find_marker_root marker parent

let find_python_project_root path =
  try find_marker_root "loom-package.json" path
  with Diagnostic.Error _ ->
    let absolute =
      if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
      else path
    in
    if not (Sys.file_exists absolute) then
      Diagnostic.raise_error (Printf.sprintf "path %S does not exist" path);
    if is_directory absolute then absolute else Filename.dirname absolute

let rec collect_source_files suffixes dir =
  Sys.readdir dir |> Array.to_list |> List.sort String.compare
  |> List.concat_map (fun name ->
         let path = Filename.concat dir name in
         if is_directory path then
           if
             List.mem name
               [
                 "_build";
                 "build";
                 ".venv";
                 ".git";
                 ".hg";
                 ".svn";
                 "node_modules";
               ]
             || (String.length name > 0 && name.[0] = '.')
           then []
           else collect_source_files suffixes path
         else if List.exists (Filename.check_suffix name) suffixes then [ path ]
         else [])

let sanitize_ident text =
  text |> lowercase
  |> String.map (fun ch ->
         if
           (ch >= 'a' && ch <= 'z')
           || (ch >= 'A' && ch <= 'Z')
           || (ch >= '0' && ch <= '9')
         then ch
         else '_')

let module_name_of_file file =
  file |> Filename.basename |> Filename.remove_extension |> sanitize_ident

let package_name_of_root root = root |> Filename.basename |> sanitize_ident

let package_name_of_python_root root =
  let marker = Filename.concat root "loom-package.json" in
  if not (Sys.file_exists marker) then package_name_of_root root
  else
    try
      let fields =
        Yojson.Safe.from_file marker |> function
        | `Assoc fields -> fields
        | _ -> []
      in
      match List.assoc_opt "name" fields with
      | Some (`String name) when String.trim name <> "" -> sanitize_ident name
      | _ -> package_name_of_root root
    with _ -> package_name_of_root root

let symbol_name module_name entry_name =
  Printf.sprintf "loom_%s_%s" module_name (sanitize_ident entry_name)

let no_workspace_symbol_name symbol_name = symbol_name ^ "_noworkspace"

let make_bundle ~source_file ~module_name ~entry_name ~lowered_sexp ~loom_entry
    ~program ~plan ~cuda_platform ~optimizations =
  let symbol_name = symbol_name module_name entry_name in
  {
    module_name;
    source_file;
    entry_name;
    symbol_name;
    workspace_symbol = symbol_name ^ "_workspace_size";
    lowered_sexp;
    loom_entry_json = LoomLambda.entry_to_string loom_entry;
    program;
    tensor_ir_json = Tensor_ir.program_to_string program;
    plan;
    kernel_plan_json =
      Shared_kernel_plan.to_string
        (Shared_kernel_plan.of_program ~optimizations program);
    cuda_plan_json = Cuda_plan.to_string plan;
    backend_analysis_json = Cuda_analysis.to_string ~program ~plan;
    optimizations;
    cuda_platform;
  }

let filter_matches filters value =
  filters = [] || List.mem (lowercase value) (List.map lowercase filters)

let load_front_entry_bundle cuda_platform optimizations file ~entry_name
    ~lowered_sexp front_entry =
  let module_name = module_name_of_file file in
  let loom_entry = Normalize.entry_of_front_ir ~optimizations front_entry in
  let program = Tensorize.program_of_entry ~optimizations loom_entry in
  let plan = Kernel_plan.of_program ~optimizations program in
  make_bundle ~source_file:file ~module_name ~entry_name ~lowered_sexp
    ~loom_entry ~program ~plan ~cuda_platform ~optimizations

let load_ocaml_entry_bundle cuda_platform optimizations file
    (entry : Ocaml_entry_scan.entry) =
  let lowered = Ocaml_raw_lambda.lower_entry entry in
  let front_entry = Ocaml_front_ir.import_entry entry in
  load_front_entry_bundle cuda_platform optimizations file
    ~entry_name:entry.name
    ~lowered_sexp:(Some (Ocaml_raw_lambda.raw_lambda_to_string lowered))
    front_entry

let load_python_entry_bundle cuda_platform optimizations file
    (entry : Python_frontend.entry_summary) =
  let front_entry = Python_frontend.import_entry file entry.name in
  load_front_entry_bundle cuda_platform optimizations file
    ~entry_name:entry.name ~lowered_sexp:None front_entry

let load_cpp_entry_bundle cuda_platform optimizations file
    (entry : Cpp_frontend.entry_summary) =
  let front_entry = Cpp_frontend.import_entry file entry.name in
  load_front_entry_bundle cuda_platform optimizations file
    ~entry_name:entry.name ~lowered_sexp:None front_entry

let discover_ocaml_entries ~cuda_platform ~optimizations ~project_root
    ~module_filters ~entry_filters =
  collect_source_files [ ".ml" ] project_root
  |> List.concat_map (fun file ->
         let module_name = module_name_of_file file in
         if not (filter_matches module_filters module_name) then []
         else
           Ocaml_entry_scan.list_entries file
           |> List.filter (fun (entry : Ocaml_entry_scan.entry) ->
                  filter_matches entry_filters entry.name)
           |> List.map (load_ocaml_entry_bundle cuda_platform optimizations file))

let discover_python_entries ~cuda_platform ~optimizations ~project_root
    ~module_filters ~entry_filters =
  collect_source_files [ ".py" ] project_root
  |> List.concat_map (fun file ->
         let module_name = module_name_of_file file in
         if not (filter_matches module_filters module_name) then []
         else
           Python_frontend.list_entries file
           |> List.filter (fun (entry : Python_frontend.entry_summary) ->
                  filter_matches entry_filters entry.name)
           |> List.map (load_python_entry_bundle cuda_platform optimizations file))

let discover_cpp_entries ~cuda_platform ~optimizations ~project_root
    ~module_filters ~entry_filters =
  collect_source_files [ ".cpp"; ".cc"; ".cxx" ] project_root
  |> List.concat_map (fun file ->
         let module_name = module_name_of_file file in
         if not (filter_matches module_filters module_name) then []
         else
           Cpp_frontend.list_entries file
           |> List.filter (fun (entry : Cpp_frontend.entry_summary) ->
                  filter_matches entry_filters entry.name)
           |> List.map (load_cpp_entry_bundle cuda_platform optimizations file))

let discover_entries ~input_kind ~cuda_platform ~optimizations ~project_root
    ~module_filters
    ~entry_filters =
  match input_kind with
  | OcamlFrontend ->
      discover_ocaml_entries ~cuda_platform ~optimizations ~project_root
        ~module_filters ~entry_filters
  | PythonFrontend ->
      discover_python_entries ~cuda_platform ~optimizations ~project_root
        ~module_filters ~entry_filters
  | CppFrontend ->
      discover_cpp_entries ~cuda_platform ~optimizations ~project_root
        ~module_filters ~entry_filters
  | AutoFrontend ->
      discover_ocaml_entries ~cuda_platform ~optimizations ~project_root
        ~module_filters ~entry_filters
      @ discover_python_entries ~cuda_platform ~optimizations ~project_root
          ~module_filters ~entry_filters
      @ discover_cpp_entries ~cuda_platform ~optimizations ~project_root
          ~module_filters ~entry_filters

let detect_cuda_arch () =
  let command =
    "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null"
  in
  let ic = Unix.open_process_in command in
  let line = try Some (input_line ic) with End_of_file -> None in
  let _ = Unix.close_process_in ic in
  match line with
  | None -> "sm_80"
  | Some value -> (
      let trimmed = String.trim value in
      match String.split_on_char '.' trimmed with
      | [ "12"; minor ] when minor <> "" ->
          (* GB10 reports compute capability 12.1, but CUDA/PyTorch target the
             binary-compatible Blackwell ISA as sm_120 for generated kernels. *)
          "sm_120"
      | [ major; minor ] when major <> "" && minor <> "" ->
          Printf.sprintf "sm_%s%s" major minor
      | _ -> "sm_80")

let cpp_float value =
  let text = Printf.sprintf "%.12g" value in
  if String.contains text '.' then text else text ^ ".0f"

let is_zero_const = function
  | Tensor_ir.SConstF32 value -> value = 0.0
  | _ -> false

let is_inline_ternary_value = function
  | Tensor_ir.SVar _ | Tensor_ir.SConstF32 _ | Tensor_ir.SConstBool _ -> true
  | Tensor_ir.SUnary _ | Tensor_ir.SBinary _ | Tensor_ir.SSelect _ -> false

let render_zero_max_ternary rendered_value =
  Printf.sprintf "((%s) > 0.0f ? %s : 0.0f)" rendered_value rendered_value

let render_zero_min_ternary rendered_value =
  Printf.sprintf "((%s) < 0.0f ? %s : 0.0f)" rendered_value rendered_value

let add_uses_terms lhs rhs value threshold =
  (lhs = value && rhs = threshold) || (lhs = threshold && rhs = value)

let is_negated_threshold threshold = function
  | Tensor_ir.SUnary (Tensor_ir.Neg, expr) -> expr = threshold
  | Tensor_ir.SBinary (Tensor_ir.Sub, zero_expr, expr) ->
      is_zero_const zero_expr && expr = threshold
  | _ -> false

let is_value_gt_threshold value threshold = function
  | Tensor_ir.SBinary (Tensor_ir.CmpGt, lhs, rhs) ->
      lhs = value && rhs = threshold
  | Tensor_ir.SBinary (Tensor_ir.CmpLt, lhs, rhs) ->
      lhs = threshold && rhs = value
  | _ -> false

let is_value_lt_neg_threshold value threshold = function
  | Tensor_ir.SBinary (Tensor_ir.CmpLt, lhs, rhs) ->
      lhs = value && is_negated_threshold threshold rhs
  | _ -> false

let soft_threshold_select = function
  | Tensor_ir.SSelect
      ( gt_condition,
        Tensor_ir.SBinary (Tensor_ir.Sub, then_value, then_threshold),
        Tensor_ir.SSelect
          ( lt_condition,
            Tensor_ir.SBinary (Tensor_ir.Add, add_lhs, add_rhs),
            zero_expr ) )
    ->
      let value = then_value in
      let threshold = then_threshold in
      if
        is_value_gt_threshold value threshold gt_condition
        && is_value_lt_neg_threshold value threshold lt_condition
        && add_uses_terms add_lhs add_rhs value threshold
        && is_zero_const zero_expr
      then Some (value, threshold)
      else None
  | Tensor_ir.SSelect
      ( lt_condition,
        Tensor_ir.SBinary (Tensor_ir.Add, add_lhs, add_rhs),
        Tensor_ir.SSelect
          ( gt_condition,
            Tensor_ir.SBinary (Tensor_ir.Sub, then_value, then_threshold),
            zero_expr ) )
    when then_value = add_lhs || then_value = add_rhs ->
      let value = then_value in
      let threshold = then_threshold in
      if
        is_value_lt_neg_threshold value threshold lt_condition
        && is_value_gt_threshold value threshold gt_condition
        && add_uses_terms add_lhs add_rhs value threshold
        && is_zero_const zero_expr
      then Some (value, threshold)
      else None
  | _ -> None

let render_soft_threshold_expr render value threshold =
  let value = render value in
  let threshold = render threshold in
  Printf.sprintf
    "(((%s) > (%s)) ? ((%s) - (%s)) : (((%s) < -(%s)) ? ((%s) + (%s)) : \
     0.0f))"
    value threshold value threshold value threshold value threshold

let render_soft_threshold_predicated_expr render value threshold =
  let value = render value in
  let threshold = render threshold in
  let fallback =
    Printf.sprintf
      "(((%s) > (%s)) ? ((%s) - (%s)) : (((%s) < -(%s)) ? ((%s) + (%s)) : \
       0.0f))"
      value threshold value threshold value threshold value threshold
  in
  Printf.sprintf
    "(((%s) >= 0.0f) ? copysignf(fmaxf(fabsf(%s) - (%s), 0.0f), %s) : \
     (%s))"
    threshold value threshold value fallback

let rec render_scalar_expr_with ?(cuda_pointwise_tail_tune = false)
    ?(cuda_plain_mul_add = false) ?(cuda_fast_divide = false) var_expr expr =
  let render =
    render_scalar_expr_with ~cuda_pointwise_tail_tune ~cuda_plain_mul_add
      ~cuda_fast_divide var_expr
  in
  match
    if cuda_pointwise_tail_tune then soft_threshold_select expr else None
  with
  | Some (value, threshold) ->
      render_soft_threshold_expr render value threshold
  | None -> (
  match expr with
  | Tensor_ir.SVar name -> var_expr name
  | Tensor_ir.SConstF32 value -> cpp_float value
  | Tensor_ir.SConstBool true -> "true"
  | Tensor_ir.SConstBool false -> "false"
  | Tensor_ir.SUnary (Tensor_ir.Neg, expr) ->
      Printf.sprintf "(-(%s))" (render expr)
  | Tensor_ir.SUnary (Tensor_ir.Sqrt, expr) ->
      Printf.sprintf "sqrtf(%s)" (render expr)
  | Tensor_ir.SUnary (Tensor_ir.Exp, expr) ->
      Printf.sprintf "expf(%s)" (render expr)
  | Tensor_ir.SUnary (Tensor_ir.Log, expr) ->
      Printf.sprintf "logf(%s)" (render expr)
  | Tensor_ir.SBinary (Tensor_ir.Add, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs), addend)
  | Tensor_ir.SBinary (Tensor_ir.Add, addend, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs)) ->
      if cuda_plain_mul_add then
        Printf.sprintf "(((%s) * (%s)) + (%s))" (render lhs) (render rhs)
          (render addend)
      else Printf.sprintf "fmaf(%s, %s, %s)" (render lhs) (render rhs)
             (render addend)
  | Tensor_ir.SBinary (Tensor_ir.Sub, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs), subtrahend) ->
      if cuda_plain_mul_add then
        Printf.sprintf "(((%s) * (%s)) - (%s))" (render lhs) (render rhs)
          (render subtrahend)
      else
        Printf.sprintf "fmaf(%s, %s, -(%s))" (render lhs) (render rhs)
          (render subtrahend)
  | Tensor_ir.SBinary (Tensor_ir.Sub, minuend, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs)) ->
      if cuda_plain_mul_add then
        Printf.sprintf "((%s) - ((%s) * (%s)))" (render minuend) (render lhs)
          (render rhs)
      else
        Printf.sprintf "fmaf(-(%s), %s, %s)" (render lhs) (render rhs)
          (render minuend)
  | Tensor_ir.SBinary (Tensor_ir.Add, lhs, rhs) ->
      Printf.sprintf "((%s) + (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.Sub, lhs, rhs) ->
      Printf.sprintf "((%s) - (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs) ->
      Printf.sprintf "((%s) * (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.Div, lhs, rhs) ->
      if cuda_fast_divide then
        Printf.sprintf "__fdividef(%s, %s)" (render lhs) (render rhs)
      else Printf.sprintf "((%s) / (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.Min, lhs, rhs) ->
      if is_zero_const lhs && is_inline_ternary_value rhs then
        render_zero_min_ternary (render rhs)
      else if is_zero_const rhs && is_inline_ternary_value lhs then
        render_zero_min_ternary (render lhs)
      else Printf.sprintf "fminf(%s, %s)" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.Max, lhs, rhs) ->
      if is_zero_const lhs && is_inline_ternary_value rhs then
        render_zero_max_ternary (render rhs)
      else if is_zero_const rhs && is_inline_ternary_value lhs then
        render_zero_max_ternary (render lhs)
      else Printf.sprintf "fmaxf(%s, %s)" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.CmpLt, lhs, rhs) ->
      Printf.sprintf "((%s) < (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.CmpLe, lhs, rhs) ->
      Printf.sprintf "((%s) <= (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.CmpGt, lhs, rhs) ->
      Printf.sprintf "((%s) > (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.CmpGe, lhs, rhs) ->
      Printf.sprintf "((%s) >= (%s))" (render lhs) (render rhs)
  | Tensor_ir.SBinary (Tensor_ir.CmpEq, lhs, rhs) ->
      Printf.sprintf "((%s) == (%s))" (render lhs) (render rhs)
  | Tensor_ir.SSelect (cond, then_expr, else_expr) ->
      Printf.sprintf "((%s) ? (%s) : (%s))" (render cond)
        (render then_expr) (render else_expr)
  )

and render_scalar_expr expr = render_scalar_expr_with (fun name -> name) expr

let rec render_scalar_expr_replacing var_expr replaced replacement expr =
  if expr = replaced then replacement
  else
    match expr with
    | Tensor_ir.SVar name -> var_expr name
    | Tensor_ir.SConstF32 value -> cpp_float value
    | Tensor_ir.SConstBool true -> "true"
    | Tensor_ir.SConstBool false -> "false"
    | Tensor_ir.SUnary (Tensor_ir.Neg, expr) ->
        Printf.sprintf "(-(%s))"
          (render_scalar_expr_replacing var_expr replaced replacement expr)
    | Tensor_ir.SUnary (Tensor_ir.Sqrt, expr) ->
        Printf.sprintf "sqrtf(%s)"
          (render_scalar_expr_replacing var_expr replaced replacement expr)
    | Tensor_ir.SUnary (Tensor_ir.Exp, expr) ->
        Printf.sprintf "expf(%s)"
          (render_scalar_expr_replacing var_expr replaced replacement expr)
    | Tensor_ir.SUnary (Tensor_ir.Log, expr) ->
        Printf.sprintf "logf(%s)"
          (render_scalar_expr_replacing var_expr replaced replacement expr)
    | Tensor_ir.SBinary (Tensor_ir.Add, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs), addend)
    | Tensor_ir.SBinary (Tensor_ir.Add, addend, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs)) ->
        Printf.sprintf "fmaf(%s, %s, %s)"
          (render_scalar_expr_replacing var_expr replaced replacement lhs)
          (render_scalar_expr_replacing var_expr replaced replacement rhs)
          (render_scalar_expr_replacing var_expr replaced replacement addend)
    | Tensor_ir.SBinary (Tensor_ir.Sub, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs), subtrahend) ->
        Printf.sprintf "fmaf(%s, %s, -(%s))"
          (render_scalar_expr_replacing var_expr replaced replacement lhs)
          (render_scalar_expr_replacing var_expr replaced replacement rhs)
          (render_scalar_expr_replacing var_expr replaced replacement subtrahend)
    | Tensor_ir.SBinary (Tensor_ir.Sub, minuend, Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs)) ->
        Printf.sprintf "fmaf(-(%s), %s, %s)"
          (render_scalar_expr_replacing var_expr replaced replacement lhs)
          (render_scalar_expr_replacing var_expr replaced replacement rhs)
          (render_scalar_expr_replacing var_expr replaced replacement minuend)
    | Tensor_ir.SBinary (op, lhs_expr, rhs_expr) ->
        let lhs =
          render_scalar_expr_replacing var_expr replaced replacement lhs_expr
        in
        let rhs =
          render_scalar_expr_replacing var_expr replaced replacement rhs_expr
        in
        (match op with
        | Tensor_ir.Add -> Printf.sprintf "((%s) + (%s))" lhs rhs
        | Tensor_ir.Sub -> Printf.sprintf "((%s) - (%s))" lhs rhs
        | Tensor_ir.Mul -> Printf.sprintf "((%s) * (%s))" lhs rhs
        | Tensor_ir.Div -> Printf.sprintf "((%s) / (%s))" lhs rhs
        | Tensor_ir.Min ->
            if is_zero_const lhs_expr && is_inline_ternary_value rhs_expr then
              render_zero_min_ternary rhs
            else if is_zero_const rhs_expr && is_inline_ternary_value lhs_expr
            then render_zero_min_ternary lhs
            else Printf.sprintf "fminf(%s, %s)" lhs rhs
        | Tensor_ir.Max ->
            if is_zero_const lhs_expr && is_inline_ternary_value rhs_expr then
              render_zero_max_ternary rhs
            else if is_zero_const rhs_expr && is_inline_ternary_value lhs_expr
            then render_zero_max_ternary lhs
            else Printf.sprintf "fmaxf(%s, %s)" lhs rhs
        | Tensor_ir.CmpLt -> Printf.sprintf "((%s) < (%s))" lhs rhs
        | Tensor_ir.CmpLe -> Printf.sprintf "((%s) <= (%s))" lhs rhs
        | Tensor_ir.CmpGt -> Printf.sprintf "((%s) > (%s))" lhs rhs
        | Tensor_ir.CmpGe -> Printf.sprintf "((%s) >= (%s))" lhs rhs
        | Tensor_ir.CmpEq -> Printf.sprintf "((%s) == (%s))" lhs rhs)
    | Tensor_ir.SSelect (cond, then_expr, else_expr) ->
        Printf.sprintf "((%s) ? (%s) : (%s))"
          (render_scalar_expr_replacing var_expr replaced replacement cond)
          (render_scalar_expr_replacing var_expr replaced replacement then_expr)
          (render_scalar_expr_replacing var_expr replaced replacement else_expr)

let render_param_decl = function
  | Tensor_ir.ScalarF32 name -> Printf.sprintf "float %s" name
  | Tensor_ir.Tensor1F32 (name, _) -> Printf.sprintf "const float* %s" name

let compare_op_to_string = function
  | Tensor_ir.CmpLt -> "<"
  | Tensor_ir.CmpLe -> "<="
  | Tensor_ir.CmpGt -> ">"
  | Tensor_ir.CmpGe -> ">="
  | Tensor_ir.CmpEq -> "=="
  | _ -> invalid_arg "not a comparison op"

let select_reuses_compared_value = function
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary
          ( ( Tensor_ir.CmpLt | Tensor_ir.CmpLe | Tensor_ir.CmpGt
            | Tensor_ir.CmpGe | Tensor_ir.CmpEq ) as op,
            lhs,
            rhs ),
        then_expr,
        else_expr )
    when lhs = then_expr ->
      Some (lhs, op, true, rhs, else_expr)
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary
          ( ( Tensor_ir.CmpLt | Tensor_ir.CmpLe | Tensor_ir.CmpGt
            | Tensor_ir.CmpGe | Tensor_ir.CmpEq ) as op,
            lhs,
            rhs ),
        then_expr,
        else_expr )
    when rhs = then_expr ->
      Some (rhs, op, false, lhs, else_expr)
  | _ -> None

let minmax_for_reused_select op reused_on_lhs =
  match op with
  | Tensor_ir.CmpLt | Tensor_ir.CmpLe ->
      Some (if reused_on_lhs then "fminf" else "fmaxf")
  | Tensor_ir.CmpGt | Tensor_ir.CmpGe ->
      Some (if reused_on_lhs then "fmaxf" else "fminf")
  | Tensor_ir.CmpEq | Tensor_ir.Add | Tensor_ir.Sub | Tensor_ir.Mul
  | Tensor_ir.Div | Tensor_ir.Min | Tensor_ir.Max ->
      None

let is_simple_scalar_expr = function
  | Tensor_ir.SVar _ | SConstF32 _ | SConstBool _ -> true
  | SUnary _ | SBinary _ | SSelect _ -> false

let rec scalar_expr_contains needle expr =
  expr = needle
  ||
  match expr with
  | Tensor_ir.SUnary (_, expr) -> scalar_expr_contains needle expr
  | SBinary (_, lhs, rhs) ->
      scalar_expr_contains needle lhs || scalar_expr_contains needle rhs
  | SSelect (cond, then_expr, else_expr) ->
      scalar_expr_contains needle cond || scalar_expr_contains needle then_expr
      || scalar_expr_contains needle else_expr
  | SVar _ | SConstF32 _ | SConstBool _ -> false

let select_reuses_compared_subexpr = function
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary
          ( ( Tensor_ir.CmpLt | Tensor_ir.CmpLe | Tensor_ir.CmpGt
            | Tensor_ir.CmpGe | Tensor_ir.CmpEq ) as op,
            lhs,
            rhs ),
        then_expr,
        else_expr )
    when (not (is_simple_scalar_expr lhs)) && scalar_expr_contains lhs then_expr ->
      Some (lhs, op, true, rhs, then_expr, else_expr)
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary
          ( ( Tensor_ir.CmpLt | Tensor_ir.CmpLe | Tensor_ir.CmpGt
            | Tensor_ir.CmpGe | Tensor_ir.CmpEq ) as op,
            lhs,
            rhs ),
        then_expr,
        else_expr )
    when (not (is_simple_scalar_expr rhs)) && scalar_expr_contains rhs then_expr ->
      Some (rhs, op, false, lhs, then_expr, else_expr)
  | _ -> None

let select_clamps_compared_value = function
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary (Tensor_ir.CmpLt, low_value, low_bound),
        low_then,
        Tensor_ir.SSelect
          ( Tensor_ir.SBinary (Tensor_ir.CmpGt, high_value, high_bound),
            high_then,
            high_else ) )
    when low_bound = low_then && high_bound = high_then
         && low_value = high_value && low_value = high_else ->
      Some (low_value, low_bound, high_bound)
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary (Tensor_ir.CmpGt, high_value, high_bound),
        high_then,
        Tensor_ir.SSelect
          ( Tensor_ir.SBinary (Tensor_ir.CmpLt, low_value, low_bound),
            low_then,
            low_else ) )
    when high_bound = high_then && low_bound = low_then
         && high_value = low_value && high_value = low_else ->
      Some (high_value, low_bound, high_bound)
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary (Tensor_ir.CmpLt, low_value, low_bound),
        low_then,
        Tensor_ir.SBinary (Tensor_ir.Min, high_value, high_bound) )
    when low_bound = low_then && low_value = high_value ->
      Some (low_value, low_bound, high_bound)
  | Tensor_ir.SSelect
      ( Tensor_ir.SBinary (Tensor_ir.CmpGt, high_value, high_bound),
        high_then,
        Tensor_ir.SBinary (Tensor_ir.Max, low_value, low_bound) )
    when high_bound = high_then && high_value = low_value ->
      Some (high_value, low_bound, high_bound)
  | _ -> None

let render_value_lines ?(declare = true) ?(scratch_suffix = "")
    ?(cuda_pointwise_tail_tune = false) ?(cuda_predicated_select = false)
    ?(cuda_eager_select_then = false) ?(cuda_plain_mul_add = false)
    ?(cuda_fast_divide = false) ~indent ~target ~render_var body =
  let target_binding = if declare then "float " ^ target else target in
  match select_clamps_compared_value body with
  | Some (value_expr, low_bound, high_bound) ->
      let render expr =
        render_scalar_expr_with ~cuda_pointwise_tail_tune ~cuda_plain_mul_add
          ~cuda_fast_divide render_var expr
      in
      let raw_target = target ^ "_unclamped" ^ scratch_suffix in
      if cuda_pointwise_tail_tune && not cuda_predicated_select then
        [
          Printf.sprintf "%s%s = %s;" indent target_binding
            (render value_expr);
          Printf.sprintf "%sif (%s < %s) %s = %s;" indent target
            (render low_bound) target (render low_bound);
          Printf.sprintf "%sif (%s > %s) %s = %s;" indent target
            (render high_bound) target (render high_bound);
        ]
      else
        [
          Printf.sprintf "%sfloat %s = %s;" indent raw_target
            (render value_expr);
          Printf.sprintf "%s%s = fminf(fmaxf(%s, %s), %s);" indent
            target_binding raw_target (render low_bound) (render high_bound);
        ]
  | None -> (
      match
        if cuda_predicated_select then soft_threshold_select body else None
      with
      | Some (value_expr, threshold_expr) ->
          let render expr =
            render_scalar_expr_with ~cuda_pointwise_tail_tune
              ~cuda_plain_mul_add ~cuda_fast_divide render_var expr
          in
          [
            Printf.sprintf "%s%s = %s;" indent target_binding
              (render_soft_threshold_predicated_expr render value_expr
                 threshold_expr);
          ]
      | None -> (
      match select_reuses_compared_subexpr body with
      | Some (reused_expr, op, reused_on_lhs, other_expr, then_expr, else_expr) ->
          let reused_target = target ^ "_guard" ^ scratch_suffix in
          let render expr =
            render_scalar_expr_replacing render_var reused_expr reused_target expr
          in
          let lhs, rhs =
            if reused_on_lhs then
              (reused_target, render other_expr)
            else (render other_expr, reused_target)
          in
          let reused_line =
            Printf.sprintf "%sfloat %s = %s;" indent reused_target
              (render_scalar_expr_with ~cuda_pointwise_tail_tune
                 ~cuda_plain_mul_add ~cuda_fast_divide render_var reused_expr)
          in
          if cuda_eager_select_then then
            let candidate_target = target ^ "_candidate" ^ scratch_suffix in
            [
              reused_line;
              Printf.sprintf "%sfloat %s = %s;" indent candidate_target
                (render then_expr);
              Printf.sprintf "%s%s = ((%s) %s (%s)) ? %s : (%s);" indent
                target_binding lhs (compare_op_to_string op) rhs
                candidate_target (render else_expr);
            ]
          else
            [
              reused_line;
              Printf.sprintf "%s%s = ((%s) %s (%s)) ? (%s) : (%s);" indent
                target_binding lhs (compare_op_to_string op) rhs
                (render then_expr) (render else_expr);
            ]
      | None -> (
      match select_reuses_compared_value body with
      | Some (reused_expr, op, reused_on_lhs, other_expr, else_expr) ->
          let render expr =
            render_scalar_expr_with ~cuda_pointwise_tail_tune
              ~cuda_plain_mul_add ~cuda_fast_divide render_var expr
          in
          let reused_target = target ^ "_selected_value" ^ scratch_suffix in
          let reused_line =
            Printf.sprintf "%sfloat %s = %s;" indent reused_target
              (render reused_expr)
          in
          begin
            match minmax_for_reused_select op reused_on_lhs with
            | Some fn when else_expr = other_expr ->
                [
                  reused_line;
                  Printf.sprintf "%s%s = %s(%s, %s);" indent target_binding fn
                    reused_target (render other_expr);
                ]
            | _ ->
                let lhs, rhs =
                  if reused_on_lhs then
                    (reused_target, render other_expr)
                  else (render other_expr, reused_target)
                in
                [
                  reused_line;
                  Printf.sprintf "%s%s = ((%s) %s (%s)) ? %s : (%s);" indent
                    target_binding lhs (compare_op_to_string op) rhs
                    reused_target (render else_expr);
                ]
          end
      | None -> (
          match body with
          | Tensor_ir.SBinary (Tensor_ir.Mul, lhs, rhs)
            when lhs = rhs && not (is_simple_scalar_expr lhs) ->
              let factor_target = target ^ "_factor" ^ scratch_suffix in
              [
                Printf.sprintf "%sfloat %s = %s;" indent factor_target
                  (render_scalar_expr_with ~cuda_pointwise_tail_tune
                     ~cuda_plain_mul_add ~cuda_fast_divide render_var lhs);
                Printf.sprintf "%s%s = %s * %s;" indent target_binding
                  factor_target factor_target;
              ]
          | _ ->
              [
                Printf.sprintf "%s%s = %s;" indent target_binding
                  (render_scalar_expr_with ~cuda_pointwise_tail_tune
                     ~cuda_plain_mul_add ~cuda_fast_divide render_var body);
              ])
      )))

let bundle_has_reduction (bundle : entry_bundle) =
  List.exists
    (function
      | Kernel_plan.Reduction _ -> true | Kernel_plan.Elementwise _ -> false)
    bundle.plan.Kernel_plan.steps

let reduction_uses_double_atomic_workspace (step : Kernel_plan.reduction_step) =
  step.execution_family = Kernel_plan.AtomicOutputReduction
  && step.reduce_kind = Tensor_ir.Sum
  && List.exists (String.equal "ratio") step.traits
  && List.exists (String.equal "clip-body") step.traits

let atomic_double_accumulator_name (step : Kernel_plan.reduction_step) =
  step.kernel_name ^ "_atomic_accum"

let bundle_uses_workspace (bundle : entry_bundle) =
  List.exists
    (function
      | Kernel_plan.Reduction step ->
          step.uses_workspace
          || reduction_uses_double_atomic_workspace step
          || not (String.equal step.output "out")
      | Kernel_plan.Elementwise step -> not (String.equal step.output "out"))
    bundle.plan.Kernel_plan.steps

let render_header_prototype (bundle : entry_bundle) =
  let base_params =
    List.map render_param_decl bundle.program.Tensor_ir.params
    @ [ "int64_t n"; "float* out" ]
  in
  let params =
    base_params @ [ "void* workspace"; "size_t workspace_size" ]
  in
  let prototypes =
    [
      Printf.sprintf "size_t %s(int64_t n);" bundle.workspace_symbol;
      Printf.sprintf "int %s(%s);" bundle.symbol_name (String.concat ", " params);
    ]
  in
  if bundle_uses_workspace bundle then prototypes
  else
    prototypes
    @ [
        Printf.sprintf "int %s(%s);"
          (no_workspace_symbol_name bundle.symbol_name)
          (String.concat ", " base_params);
      ]

let min_block_size default buckets =
  List.fold_left
    (fun acc (bucket : Kernel_plan.launch_bucket) -> min acc bucket.block_size)
    default buckets

let result_byte_expr (bundle : entry_bundle) =
  match bundle.program.Tensor_ir.result with
  | Tensor_ir.TensorResult _ -> "static_cast<size_t>(n) * sizeof(float)"
  | Tensor_ir.ScalarResult _ -> "sizeof(float)"

let shared_reduction_lines reduce_kind =
  let combine target source =
    match reduce_kind with
    | Tensor_ir.Sum ->
        Printf.sprintf "%s += %s;" target source
    | Tensor_ir.MaxReduce ->
        Printf.sprintf "%s = fmaxf(%s, %s);" target target source
  in
  [
    "  shared[threadIdx.x] = accum;";
    "  __syncthreads();";
    "  for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {";
    "    if (threadIdx.x < offset) {";
    Printf.sprintf "      %s"
      (combine "shared[threadIdx.x]" "shared[threadIdx.x + offset]");
    "    }";
    "    __syncthreads();";
    "  }";
    "  if (threadIdx.x < 32) {";
    "    volatile float* vshared = shared;";
    Printf.sprintf "    if (blockDim.x >= 64) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 32]");
    Printf.sprintf "    if (blockDim.x >= 32) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 16]");
    Printf.sprintf "    if (blockDim.x >= 16) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 8]");
    Printf.sprintf "    if (blockDim.x >= 8) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 4]");
    Printf.sprintf "    if (blockDim.x >= 4) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 2]");
    Printf.sprintf "    if (blockDim.x >= 2) %s"
      (combine "vshared[threadIdx.x]" "vshared[threadIdx.x + 1]");
    "  }";
    "  accum = shared[threadIdx.x];";
  ]

let shuffle_reduction_lines reduce_kind =
  [
    (match reduce_kind with
    | Tensor_ir.Sum -> "  accum = loom_block_reduce_sum(accum);"
    | Tensor_ir.MaxReduce -> "  accum = loom_block_reduce_max(accum);");
  ]

let reduction_lines (step : Kernel_plan.reduction_step) =
  if
    List.exists (String.equal "shuffle-reduce") step.traits
    || String.equal step.stage_layout "cuda-shuffle"
  then shuffle_reduction_lines step.reduce_kind
  else shared_reduction_lines step.reduce_kind

let reduction_uses_shuffle (step : Kernel_plan.reduction_step) =
  List.exists (String.equal "shuffle-reduce") step.traits
  || String.equal step.stage_layout "cuda-shuffle"

let reduction_uses_unroll4 (step : Kernel_plan.reduction_step) =
  step.reduce_kind = Tensor_ir.Sum
  && List.mem step.reduction_family
       [
         "mapped";
         "norm-square";
         "affine-norm-square";
         "dot-product";
         "delta-square";
         "weighted-product";
         "ratio";
         "branchy";
         "robust";
         "clipped-robust";
       ]

let reduction_prefers_capped_stage_blocks (step : Kernel_plan.reduction_step) =
  List.exists
    (fun trait ->
      String.equal trait "branch-body" || String.equal trait "ratio"
      || String.equal trait "clipped")
    step.traits
  || List.mem step.reduction_family [ "branchy"; "ratio"; "robust"; "clipped-robust" ]

let reduction_uses_mid_stage_cap optimizations (step : Kernel_plan.reduction_step)
    =
  if
    Optimizations.enabled optimizations Optimizations.CudaReductionTailTune
    && reduction_prefers_capped_stage_blocks step
  then true
  else false

let render_reduction_first_kernel (step : Kernel_plan.reduction_step) =
  let atomic_output_type =
    if reduction_uses_double_atomic_workspace step then "double" else "float"
  in
  let init, accum_update =
    match step.reduce_kind with
    | Tensor_ir.Sum ->
        ( "0.0f",
          "accum += value;" )
    | Tensor_ir.MaxReduce ->
        ( "-INFINITY",
          "accum = fmaxf(accum, value);" )
  in
  let params =
    match (step.execution_family, step.source) with
    | Kernel_plan.AtomicOutputReduction, Kernel_plan.PlainInput _ ->
        [
          Printf.sprintf "%s* __restrict__ out" atomic_output_type;
          "const float* __restrict__ input";
          "int64_t n";
        ]
    | Kernel_plan.AtomicOutputReduction, Kernel_plan.MappedInput { inputs; scalar_params; _ } ->
        [ Printf.sprintf "%s* __restrict__ out" atomic_output_type; "int64_t n" ]
        @ List.map
            (fun (name, _) ->
              Printf.sprintf "const float* __restrict__ input_%s" name)
            inputs
        @ List.map (fun name -> Printf.sprintf "float %s" name) scalar_params
    | _, Kernel_plan.PlainInput _ ->
        [
          "const float* __restrict__ input";
          "float* __restrict__ partial";
          "int64_t n";
        ]
    | _, Kernel_plan.MappedInput { inputs; scalar_params; _ } ->
        [ "float* __restrict__ partial"; "int64_t n" ]
        @ List.map
            (fun (name, _) ->
              Printf.sprintf "const float* __restrict__ input_%s" name)
            inputs
        @ List.map (fun name -> Printf.sprintf "float %s" name) scalar_params
  in
  let loop_lines =
    match step.source with
    | Kernel_plan.PlainInput _ ->
        if reduction_uses_unroll4 step then
          [
            "  for (; idx + (3 * stride) < n; idx += 4 * stride) {";
            "    float value = input[idx];";
            Printf.sprintf "    %s" accum_update;
            "    value = input[idx + stride];";
            Printf.sprintf "    %s" accum_update;
            "    value = input[idx + (2 * stride)];";
            Printf.sprintf "    %s" accum_update;
            "    value = input[idx + (3 * stride)];";
            Printf.sprintf "    %s" accum_update;
            "  }";
            "  for (; idx < n; idx += stride) {";
            "    float value = input[idx];";
            Printf.sprintf "    %s" accum_update;
            "  }";
          ]
        else
          [
            "  for (; idx < n; idx += stride) {";
            "    float value = input[idx];";
            Printf.sprintf "    %s" accum_update;
            "  }";
          ]
    | Kernel_plan.MappedInput { inputs; body; _ } ->
        let scalar_input_lines index_expr suffix =
          List.map
            (fun (name, _) ->
              Printf.sprintf "    float %s%s = input_%s[%s];" name suffix name
                index_expr)
            inputs
        in
        let generic_value_lines index_expr suffix =
          let declare_value = String.equal suffix "_0" || String.equal suffix "" in
          scalar_input_lines index_expr suffix
          @ render_value_lines ~declare:declare_value ~scratch_suffix:suffix
              ~indent:"    " ~target:"value"
              ~render_var:(fun name ->
                if
                  List.exists
                    (fun (input_name, _) -> String.equal input_name name)
                    inputs
                then name ^ suffix
                else name)
              body
          @ [ Printf.sprintf "    %s" accum_update ]
        in
        let specialized_value_lines index_expr suffix =
          let value_binding =
            if String.equal suffix "_0" || String.equal suffix "" then
              "float value"
            else "value"
          in
          let weighted_accum_lines xi_name yi_name =
            scalar_input_lines index_expr suffix
            @ [
                Printf.sprintf
                  "    accum = fmaf((%s%s * weight), %s%s, accum);" xi_name
                  suffix yi_name suffix;
              ]
          in
          let mixed_weighted_accum_lines xi_name yi_name =
            scalar_input_lines index_expr suffix
            @ [
                Printf.sprintf
                  "    float transformed%s = fmaf(scale, %s%s, bias);" suffix
                  xi_name suffix;
                Printf.sprintf
                  "    accum = fmaf((transformed%s * weight), %s%s, accum);"
                  suffix yi_name suffix;
              ]
          in
          match (step.reduction_family, inputs) with
          | ("norm-square", [ (xi_name, _) ]) ->
              scalar_input_lines index_expr suffix
              @ [
                  Printf.sprintf
                    "    accum = fmaf(%s%s, %s%s, accum);" xi_name suffix
                    xi_name suffix;
                ]
          | ("affine-norm-square", [ (xi_name, _) ]) ->
              scalar_input_lines index_expr suffix
              @ [
                  Printf.sprintf
                    "    float transformed%s = fmaf(scale, %s%s, bias);" suffix
                    xi_name suffix;
                  Printf.sprintf
                    "    accum = fmaf(transformed%s, transformed%s, accum);"
                    suffix suffix;
                ]
          | ("dot-product", [ (xi_name, _); (yi_name, _) ]) ->
              scalar_input_lines index_expr suffix
              @ [
                  Printf.sprintf
                    "    accum = fmaf(%s%s, %s%s, accum);" xi_name suffix
                    yi_name suffix;
                ]
          | ("delta-square", [ (xi_name, _); (yi_name, _) ]) ->
              scalar_input_lines index_expr suffix
              @ [
                  Printf.sprintf "    float diff%s = %s%s - %s%s;" suffix
                    xi_name suffix yi_name suffix;
                  Printf.sprintf
                    "    accum = fmaf(diff%s, diff%s, accum);" suffix suffix;
                ]
          | ("weighted-product", [ (xi_name, _); (yi_name, _) ]) -> (
              match body with
              | Tensor_ir.SBinary
                  ( Tensor_ir.Mul,
                    Tensor_ir.SBinary
                      ( Tensor_ir.Mul,
                        Tensor_ir.SVar lhs_name,
                        Tensor_ir.SVar rhs_name ),
                    Tensor_ir.SVar weight_name )
                when String.equal lhs_name xi_name
                     && String.equal rhs_name yi_name
                     && String.equal weight_name "weight" ->
                  weighted_accum_lines xi_name yi_name
              | Tensor_ir.SBinary
                  ( Tensor_ir.Mul,
                    Tensor_ir.SBinary
                      ( Tensor_ir.Mul,
                        Tensor_ir.SBinary
                          ( Tensor_ir.Add,
                            Tensor_ir.SBinary
                              ( Tensor_ir.Mul,
                                Tensor_ir.SVar scale_name,
                                Tensor_ir.SVar lhs_name ),
                            Tensor_ir.SVar bias_name ),
                        Tensor_ir.SVar rhs_name ),
                    Tensor_ir.SVar weight_name )
                when String.equal scale_name "scale"
                     && String.equal lhs_name xi_name
                     && String.equal bias_name "bias"
                     && String.equal rhs_name yi_name
                     && String.equal weight_name "weight" ->
                  mixed_weighted_accum_lines xi_name yi_name
              | _ -> generic_value_lines index_expr suffix)
          | ( "robust",
              [ (xi_name, _); (yi_name, _) ] ) ->
              scalar_input_lines index_expr suffix
              @ [
                Printf.sprintf "    float diff%s = %s%s - %s%s;" suffix xi_name suffix yi_name suffix;
                Printf.sprintf
                  "    float abs_diff%s = fabsf(diff%s);"
                  suffix suffix;
                Printf.sprintf
                  "    float quadratic%s = 0.5f * diff%s * diff%s;" suffix
                  suffix suffix;
                Printf.sprintf
                  "    float linear%s = delta * (abs_diff%s - (0.5f * delta));"
                  suffix suffix;
                Printf.sprintf
                  "    %s = abs_diff%s > delta ? linear%s : quadratic%s;"
                  value_binding suffix suffix suffix;
                Printf.sprintf "    %s" accum_update;
              ]
          | ( "clipped-robust",
              [ (xi_name, _); (yi_name, _) ] ) ->
              scalar_input_lines index_expr suffix
              @ [
                Printf.sprintf "    float diff%s = %s%s - %s%s;" suffix xi_name suffix yi_name suffix;
                Printf.sprintf
                  "    float abs_diff%s = fabsf(diff%s);"
                  suffix suffix;
                Printf.sprintf
                  "    float quadratic%s = 0.5f * diff%s * diff%s;"
                  suffix suffix suffix;
                Printf.sprintf
                  "    float linear%s = delta * (abs_diff%s - (0.5f * delta));"
                  suffix suffix;
                Printf.sprintf
                  "    float huber_value%s = abs_diff%s > delta ? linear%s : quadratic%s;"
                  suffix suffix suffix suffix;
                Printf.sprintf
                  "    %s = huber_value%s > cap ? cap : huber_value%s;"
                  value_binding suffix suffix;
                Printf.sprintf "    %s" accum_update;
              ]
          | _ -> generic_value_lines index_expr suffix
        in
        if reduction_uses_unroll4 step then
          [ "  for (; idx + (3 * stride) < n; idx += 4 * stride) {" ]
          @ specialized_value_lines "idx" "_0"
          @ specialized_value_lines "idx + stride" "_1"
          @ specialized_value_lines "idx + (2 * stride)" "_2"
          @ specialized_value_lines "idx + (3 * stride)" "_3"
          @ [ "  }"; "  for (; idx < n; idx += stride) {" ]
          @ specialized_value_lines "idx" ""
          @ [ "  }" ]
        else
          [ "  for (; idx < n; idx += stride) {" ]
          @ specialized_value_lines "idx" ""
          @ [ "  }" ]
  in
  let finalize_lines =
    match step.execution_family with
    | Kernel_plan.AtomicOutputReduction ->
        [
          (if reduction_uses_double_atomic_workspace step then
             "  if (threadIdx.x == 0) atomicAdd(out, static_cast<double>(accum));"
           else "  if (threadIdx.x == 0) atomicAdd(out, accum);");
          "}";
          "";
        ]
    | _ ->
        [
          "  if (threadIdx.x == 0) partial[blockIdx.x] = accum;";
          "}";
          "";
        ]
  in
  String.concat "\n"
    ([
       Printf.sprintf "__global__ void %s(%s) {" step.kernel_name
         (String.concat ", " params);
       (if reduction_uses_shuffle step then "" else "  __shared__ float shared[1024];");
       "  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + \
        threadIdx.x;";
       "  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;";
       Printf.sprintf "  float accum = %s;" init;
     ]
    @ loop_lines
    @ reduction_lines step
    @ finalize_lines)

let render_atomic_double_finalize_kernel (step : Kernel_plan.reduction_step) =
  if not (reduction_uses_double_atomic_workspace step) then ""
  else
    String.concat "\n"
      [
        Printf.sprintf
          "__global__ void %s_finalize(float* __restrict__ out, const double* __restrict__ accum) {"
          step.kernel_name;
        "  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = static_cast<float>(accum[0]);";
        "}";
        "";
      ]

let render_reduction_combine_kernel (step : Kernel_plan.reduction_step) =
  let init, accum_update =
    match step.reduce_kind with
    | Tensor_ir.Sum ->
        ( "0.0f",
          "accum += input[idx];" )
    | Tensor_ir.MaxReduce ->
        ( "-INFINITY",
          "accum = fmaxf(accum, input[idx]);" )
  in
  String.concat "\n"
    ([
      Printf.sprintf
        "__global__ void %s(const float* __restrict__ input, float* __restrict__ partial, int64_t n) {"
        step.combine_kernel_name;
      (if reduction_uses_shuffle step then "" else "  __shared__ float shared[1024];");
      "  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + \
       threadIdx.x;";
      Printf.sprintf "  float accum = %s;" init;
      "  for (; idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {";
      Printf.sprintf "    %s" accum_update;
      "  }";
    ]
    @ reduction_lines step
    @ [
      "  if (threadIdx.x == 0) partial[blockIdx.x] = accum;";
      "}";
      "";
    ])

let render_elementwise_kernel ?(single_pass = false) optimizations
    (step : Kernel_plan.elementwise_step) body =
  let cuda_pointwise_tail_tune =
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
  in
  let cuda_predicated_select =
    Optimizations.enabled optimizations Optimizations.CudaPointwisePredicatedSelect
  in
  let cuda_selected_tail_plan =
    Optimizations.enabled optimizations Optimizations.CudaPointwiseSelectedTailPlan
  in
  let has_step_trait name =
    List.exists (String.equal name) step.Kernel_plan.traits
  in
  let affine_threshold_clamp =
    has_step_trait "affine" && has_step_trait "threshold"
    && List.length step.inputs = 1
  in
  let affine_clip_threshold =
    affine_threshold_clamp && has_step_trait "clip"
  in
  let cuda_predicated_select_for_body =
    cuda_predicated_select
    && not (cuda_selected_tail_plan && affine_clip_threshold)
    && (affine_threshold_clamp
       || not (has_step_trait "threshold" && not (has_step_trait "clip")))
  in
  let cuda_plain_mul_add =
    cuda_selected_tail_plan
    && ((has_step_trait "affine-vector-update" && not (has_step_trait "ratio"))
       || (has_step_trait "affine" && has_step_trait "clip"
          && has_step_trait "threshold"))
  in
  let cuda_eager_select_then =
    cuda_pointwise_tail_tune && has_step_trait "mixed-filter-ratio"
  in
  let cuda_fast_divide =
    cuda_selected_tail_plan
    && has_step_trait "ratio-book"
    && not (has_step_trait "branch")
  in
  let params =
    [ "float* __restrict__ out"; "int64_t n" ]
    @ List.map
        (fun (name, _) ->
          Printf.sprintf "const float* __restrict__ input_%s" name)
        step.Kernel_plan.inputs
    @ List.map (fun name -> Printf.sprintf "float %s" name) step.scalar_params
  in
  let value_lines =
    render_value_lines ~cuda_pointwise_tail_tune ~indent:"    " ~target:"value"
      ~cuda_predicated_select:cuda_predicated_select_for_body
      ~cuda_eager_select_then ~cuda_plain_mul_add ~cuda_fast_divide
      ~render_var:(fun name -> name) body
  in
  let loop_header, loop_footer =
    if single_pass then ([ "  if (idx < n) {" ], [ "  }" ])
    else
      ( [
          "  for (; idx < n; idx += static_cast<int64_t>(blockDim.x) * \
           gridDim.x) {";
        ],
        [ "  }" ] )
  in
  String.concat "\n"
    ([
       Printf.sprintf "__global__ void %s(%s) {" step.kernel_name
         (String.concat ", " params);
       "  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + \
        threadIdx.x;";
     ]
    @ loop_header
    @ List.map
        (fun (name, _) ->
          Printf.sprintf "    float %s = input_%s[idx];" name name)
        step.inputs
    @ value_lines
    @ [ "    out[idx] = value;" ]
    @ loop_footer
    @ [ "}"; "" ])

let scalar_fallback_step (step : Kernel_plan.elementwise_step) =
  { step with Kernel_plan.kernel_name = step.Kernel_plan.kernel_name ^ "_scalar" }

let scalar_single_pass_step (step : Kernel_plan.elementwise_step) =
  { step with Kernel_plan.kernel_name = step.Kernel_plan.kernel_name ^ "_scalar_single" }

let single_pass_step (step : Kernel_plan.elementwise_step) =
  { step with Kernel_plan.kernel_name = step.Kernel_plan.kernel_name ^ "_single" }

let vector_single_pass_step (step : Kernel_plan.elementwise_step) =
  { step with Kernel_plan.kernel_name = step.Kernel_plan.kernel_name ^ "_vector_single" }

let render_vectorized_elementwise_kernel ?(single_pass = false) optimizations
    (step : Kernel_plan.elementwise_step) body =
  let cuda_pointwise_tail_tune =
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
  in
  let cuda_predicated_select =
    Optimizations.enabled optimizations Optimizations.CudaPointwisePredicatedSelect
  in
  let cuda_selected_tail_plan =
    Optimizations.enabled optimizations Optimizations.CudaPointwiseSelectedTailPlan
  in
  let has_step_trait name =
    List.exists (String.equal name) step.Kernel_plan.traits
  in
  let affine_threshold_clamp =
    has_step_trait "affine" && has_step_trait "threshold"
    && List.length step.inputs = 1
  in
  let affine_clip_threshold =
    affine_threshold_clamp && has_step_trait "clip"
  in
  let cuda_predicated_select_for_body =
    cuda_predicated_select
    && not (cuda_selected_tail_plan && affine_clip_threshold)
    && (affine_threshold_clamp
       || not (has_step_trait "threshold" && not (has_step_trait "clip")))
  in
  let cuda_plain_mul_add =
    cuda_selected_tail_plan
    && ((has_step_trait "affine-vector-update" && not (has_step_trait "ratio"))
       || (has_step_trait "affine" && has_step_trait "clip"
          && has_step_trait "threshold"))
  in
  let cuda_eager_select_then =
    cuda_pointwise_tail_tune && has_step_trait "mixed-filter-ratio"
  in
  let cuda_fast_divide =
    cuda_selected_tail_plan
    && has_step_trait "ratio-book"
    && not (has_step_trait "branch")
  in
  let cuda_guarded_vector_tail =
    cuda_selected_tail_plan
    && ((has_step_trait "affine-vector-update" && not (has_step_trait "ratio"))
       || has_step_trait "simple-activation")
  in
  let params =
    [ "float* __restrict__ out"; "int64_t n" ]
    @ List.map
        (fun (name, _) ->
          Printf.sprintf "const float* __restrict__ input_%s" name)
        step.Kernel_plan.inputs
    @ List.map (fun name -> Printf.sprintf "float %s" name) step.scalar_params
  in
  let vector_name name = name ^ "_vec" in
  let lane_field = function
    | 0 -> "x"
    | 1 -> "y"
    | 2 -> "z"
    | _ -> "w"
  in
  let vector_value_expr lane =
    render_scalar_expr_with
      ~cuda_pointwise_tail_tune ~cuda_plain_mul_add ~cuda_fast_divide
      (fun name ->
        if List.exists (fun (input_name, _) -> String.equal input_name name) step.inputs
        then Printf.sprintf "%s.%s" (vector_name name) (lane_field lane)
        else name)
      body
  in
  let vector_lane_lines lane =
    let target = Printf.sprintf "value_%d" lane in
    render_value_lines ~indent:"    " ~target
      ~cuda_pointwise_tail_tune
      ~cuda_predicated_select:cuda_predicated_select_for_body
      ~cuda_eager_select_then ~cuda_plain_mul_add ~cuda_fast_divide
      ~render_var:(fun name ->
        if List.exists (fun (input_name, _) -> String.equal input_name name) step.inputs
        then Printf.sprintf "%s.%s" (vector_name name) (lane_field lane)
        else name)
      body
  in
  let scalar_fallback_value_lines indent =
    render_value_lines ~indent ~target:"value"
      ~cuda_pointwise_tail_tune
      ~cuda_predicated_select:cuda_predicated_select_for_body
      ~cuda_eager_select_then ~cuda_plain_mul_add ~cuda_fast_divide
      ~render_var:(fun name -> name) body
  in
  let vector_result_lines =
    if
      select_clamps_compared_value body = None
      && select_reuses_compared_value body = None
      && select_reuses_compared_subexpr body = None
      && (not cuda_predicated_select_for_body || soft_threshold_select body = None)
    then
      [
        Printf.sprintf "    float4 result = {%s, %s, %s, %s};"
          (vector_value_expr 0) (vector_value_expr 1) (vector_value_expr 2)
          (vector_value_expr 3);
      ]
    else
      List.concat
        [ vector_lane_lines 0; vector_lane_lines 1; vector_lane_lines 2; vector_lane_lines 3 ]
      @ [ "    float4 result = {value_0, value_1, value_2, value_3};" ]
  in
  let guard_empty_tail =
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && has_step_trait "filter-or-book"
    && not (has_step_trait "ratio")
  in
  let tail_lines =
    if guard_empty_tail then
      (if single_pass then
         [
           "  int64_t tail_start = vector_count * 4;";
           "  if (tail_start < n) {";
           "    int64_t idx = tail_start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;";
           "    if (idx < n) {";
         ]
       else
         [
           "  int64_t tail_start = vector_count * 4;";
           "  if (tail_start < n) {";
           "    int64_t idx = tail_start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;";
           "    for (; idx < n; idx += vector_stride) {";
         ])
      @ List.map
          (fun (name, _) ->
            Printf.sprintf "      float %s = input_%s[idx];" name name)
          step.inputs
      @ scalar_fallback_value_lines "      "
      @ [
          "      out[idx] = value;";
          "    }";
          "  }";
        ]
    else
      (if single_pass then
         [
           "  int64_t idx = (vector_count * 4) + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;";
           "  if (idx < n) {";
         ]
       else
         [
           "  int64_t idx = (vector_count * 4) + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;";
           "  for (; idx < n; idx += vector_stride) {";
         ])
      @ List.map
          (fun (name, _) ->
            Printf.sprintf "    float %s = input_%s[idx];" name name)
          step.inputs
      @ scalar_fallback_value_lines "    "
      @ [
          "    out[idx] = value;";
          "  }";
        ]
  in
  let vector_load_lines indent =
    List.map
      (fun (name, _) ->
        Printf.sprintf
          "%sfloat4 %s = reinterpret_cast<const float4*>(input_%s)[vector_idx];"
          indent (vector_name name) name)
      step.inputs
  in
  let guarded_tail_lines =
    [
      "    } else {";
      "      for (int lane = 0; lane < 4; ++lane) {";
      "        int64_t idx = base + lane;";
      "        if (idx < n) {";
    ]
    @ List.map
        (fun (name, _) ->
          Printf.sprintf "          float %s = input_%s[idx];" name name)
        step.inputs
    @ scalar_fallback_value_lines "          "
    @ [
        "          out[idx] = value;";
        "        }";
        "      }";
        "    }";
      ]
  in
  let loop_lines =
    if cuda_guarded_vector_tail then
      [
        "  int64_t vector_count = (n + 3) / 4;";
        (if single_pass then "  if (vector_idx < vector_count) {"
         else "  for (; vector_idx < vector_count; vector_idx += vector_stride) {");
        "    int64_t base = vector_idx * 4;";
        "    if (base + 3 < n) {";
      ]
      @ vector_load_lines "      "
      @ List.map (fun line -> "  " ^ line) vector_result_lines
      @ [
          "      reinterpret_cast<float4*>(out)[vector_idx] = result;";
        ]
      @ guarded_tail_lines
      @ [ "  }" ]
    else
      [
        "  int64_t vector_count = n / 4;";
        (if single_pass then "  if (vector_idx < vector_count) {"
         else "  for (; vector_idx < vector_count; vector_idx += vector_stride) {");
      ]
      @ vector_load_lines "    "
      @ vector_result_lines
      @ [
          "    reinterpret_cast<float4*>(out)[vector_idx] = result;";
          "  }";
        ]
      @ tail_lines
  in
  String.concat "\n"
    ([
       Printf.sprintf "__global__ void %s(%s) {" step.kernel_name
         (String.concat ", " params);
       "  int64_t vector_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;";
     ]
    @ (if single_pass then []
       else
         [
           "  int64_t vector_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;";
         ])
    @ loop_lines @ [ "}"; "" ])

let body_of_node (bundle : entry_bundle) node_id =
  match
    List.find_opt
      (function
        | Tensor_ir.Elementwise1D { id; _ } -> id = node_id
        | Tensor_ir.Reduce1D _ -> false)
      bundle.program.Tensor_ir.nodes
  with
  | Some (Tensor_ir.Elementwise1D { body; _ }) -> body
  | _ ->
      Diagnostic.raise_error
        "internal error: missing elementwise node for CUDA codegen"

let render_step_kernels (bundle : entry_bundle) =
  bundle.plan.Kernel_plan.steps
  |> List.filter_map (function
       | Kernel_plan.Elementwise step ->
           let body = body_of_node bundle step.node_id in
           Some
             (match step.pointwise_family with
             | Kernel_plan.VectorizedPointwiseKernel ->
                 render_elementwise_kernel bundle.optimizations
                   (scalar_fallback_step step) body
                 ^ render_elementwise_kernel bundle.optimizations
                     ~single_pass:true (scalar_single_pass_step step) body
                 ^ render_vectorized_elementwise_kernel bundle.optimizations
                     ~single_pass:true (vector_single_pass_step step) body
                 ^ render_vectorized_elementwise_kernel bundle.optimizations step
                     body
             | Kernel_plan.GenericPointwiseKernel ->
                 render_elementwise_kernel bundle.optimizations step body
                 ^ render_elementwise_kernel bundle.optimizations
                     ~single_pass:true (single_pass_step step) body
             | Kernel_plan.SmallNPointwiseKernel ->
                 render_elementwise_kernel bundle.optimizations step body)
       | Kernel_plan.Reduction step ->
           Some
             (render_reduction_first_kernel step
             ^ render_atomic_double_finalize_kernel step
             ^
             match step.combine_family with
             | Kernel_plan.NoCombineKernel -> ""
             | Kernel_plan.SharedTreeCombine ->
                 render_reduction_combine_kernel step))

let render_workspace_function (bundle : entry_bundle) =
  let lines =
    ref
      [
        Printf.sprintf "extern \"C\" size_t %s(int64_t n) {"
          bundle.workspace_symbol;
      ]
  in
  let add line = lines := !lines @ [ line ] in
  add "  if (n <= 0) return 0;";
  add "  size_t bytes = 0;";
  List.iter
    (function
        | Kernel_plan.Elementwise step when not (String.equal step.output "out") ->
          add "  bytes = loom_align_up(bytes, 256);";
          add "  bytes += static_cast<size_t>(n) * sizeof(float);"
        | Kernel_plan.Reduction step ->
            if reduction_uses_double_atomic_workspace step then (
              add "  bytes = loom_align_up(bytes, 256);";
              add "  bytes += sizeof(double);")
            else if step.uses_workspace then
              let partials =
                Printf.sprintf "loom_choose_block_count_for(n, %d)"
                  (min_block_size step.block_size step.launch_buckets)
              in
              add "  bytes = loom_align_up(bytes, 256);";
              add
                (Printf.sprintf
                   "  bytes += static_cast<size_t>(%s) * sizeof(float);" partials);
              add "  bytes = loom_align_up(bytes, 256);";
              add
                (Printf.sprintf
                   "  bytes += static_cast<size_t>(%s) * sizeof(float);" partials)
            else if not (String.equal step.output "out") then (
              add "  bytes = loom_align_up(bytes, 256);";
              add "  bytes += sizeof(float);")
        | Kernel_plan.Elementwise _ -> ())
    bundle.plan.Kernel_plan.steps;
  add "  return bytes;";
  add "}";
  add "";
  String.concat "\n" !lines

let lookup_result_ref (bundle : entry_bundle) =
  match bundle.program.Tensor_ir.result with
  | Tensor_ir.TensorResult value | Tensor_ir.ScalarResult value -> value

let render_copy_from_result (bundle : entry_bundle) =
  match lookup_result_ref bundle with
  | Tensor_ir.NodeRef _ -> []
  | Tensor_ir.ParamRef name ->
      [
        "  {";
        Printf.sprintf
          "    cudaError_t status = cudaMemcpy(out, %s, %s, \
           cudaMemcpyDeviceToDevice);"
          name (result_byte_expr bundle);
        "    if (status != cudaSuccess) return LOOM_STATUS_CUDA_ERROR;";
        "  }";
      ]

let step_has_trait name (step : Kernel_plan.elementwise_step) =
  List.exists (String.equal name) step.Kernel_plan.traits

let step_prefers_small_vector_tail (step : Kernel_plan.elementwise_step) =
  step_has_trait "simple-activation" step || step_has_trait "threshold" step
  || step_has_trait "ratio-book" step
  || step_has_trait "mixed-filter-ratio" step
  || (step_has_trait "affine-vector-update" step
     && not (step_has_trait "ratio" step))
  || (step_has_trait "filter-or-book" step && not (step_has_trait "ratio" step))

let step_prefers_deferred_vector_tail (step : Kernel_plan.elementwise_step) =
  step_has_trait "threshold" step
  || (step_has_trait "filter-or-book" step && not (step_has_trait "ratio" step))

let pointwise_vector_threshold_symbol optimizations entry_name
    (step : Kernel_plan.elementwise_step) =
  ignore entry_name;
  if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && Optimizations.enabled optimizations
         Optimizations.CudaPointwiseSelectedTailPlan
    && ((step_has_trait "affine" step
          && (step_has_trait "clip" step || step_has_trait "threshold" step)
          && List.length step.inputs = 1)
       || step_has_trait "ratio-book" step)
  then "0"
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_prefers_deferred_vector_tail step
  then "loom_kVectorPointwiseThreshold"
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_prefers_small_vector_tail step
  then "loom_kSmallVectorPointwiseThreshold"
  else if step_prefers_small_vector_tail step then "loom_kSmallVectorPointwiseThreshold"
  else "loom_kVectorPointwiseThreshold"

let step_is_affine_clip (step : Kernel_plan.elementwise_step) =
  step_has_trait "affine" step
  && (step_has_trait "clip" step || step_has_trait "threshold" step)
  && List.length step.inputs = 1

let pointwise_prefers_vector_single_pass optimizations
    (step : Kernel_plan.elementwise_step) =
  Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
  && Optimizations.enabled optimizations
       Optimizations.CudaPointwiseSelectedTailPlan
  && step_is_affine_clip step

let pointwise_vector_upper_bound optimizations (step : Kernel_plan.elementwise_step) =
  if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseMediumPlan
    && step_is_affine_clip step
  then Some 2097152
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_has_trait "filter-or-book" step
    && not (step_has_trait "ratio" step)
  then Some 2097152
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_has_trait "ratio-book" step
  then Some 2097152
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_has_trait "mixed-filter-ratio" step
  then Some 4194304
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && Optimizations.enabled optimizations
         Optimizations.CudaPointwiseSelectedTailPlan
    && (step_has_trait "simple-activation" step
       || (step_has_trait "affine-vector-update" step
          && not (step_has_trait "ratio" step)))
  then Some 4194304
  else if
    Optimizations.enabled optimizations Optimizations.CudaPointwiseTailTune
    && step_has_trait "threshold" step
    && not (step_has_trait "clip" step)
  then Some 4194304
  else None

let pointwise_prefers_hft_base_block_cap optimizations
    (step : Kernel_plan.elementwise_step) =
  (Optimizations.enabled optimizations Optimizations.CudaBookFilterRegisterPlan
  && (step_has_trait "filter-or-book" step || step_has_trait "ratio-book" step))

let pointwise_prefers_medium_scalar_loop optimizations
    (step : Kernel_plan.elementwise_step) =
  Optimizations.enabled optimizations Optimizations.CudaPointwiseMediumPlan
  && step_is_affine_clip step

let render_wrapper ?(no_workspace = false) (bundle : entry_bundle) =
  let params =
    List.map render_param_decl bundle.program.Tensor_ir.params
    @
    if no_workspace then [ "int64_t n"; "float* out" ]
    else [ "int64_t n"; "float* out"; "void* workspace"; "size_t workspace_size" ]
  in
  let symbol_name =
    if no_workspace then no_workspace_symbol_name bundle.symbol_name
    else bundle.symbol_name
  in
  let fast_async_no_workspace =
    no_workspace
    || (Optimizations.enabled bundle.optimizations
          Optimizations.CudaAsyncWrapperReturn
       && not (bundle_uses_workspace bundle))
  in
  let lines =
    ref
      [
        Printf.sprintf "extern \"C\" int %s(%s) {" symbol_name
          (String.concat ", " params);
      ]
  in
  let add line = lines := !lines @ [ line ] in
  let pick_block_lines indent current_n block_size launch_buckets =
    let bucket_lines =
      launch_buckets
      |> List.mapi (fun index (bucket : Kernel_plan.launch_bucket) ->
             let keyword = if index = 0 then "if" else "else if" in
             Printf.sprintf "%s%s (%s <= %d) block_size = %d;" indent keyword
               current_n bucket.max_n bucket.block_size)
    in
    [ Printf.sprintf "%sint block_size = %d;" indent block_size ] @ bucket_lines
  in
  if fast_async_no_workspace then add "  if (n <= 0) return LOOM_STATUS_OK;"
  else (
    if bundle_has_reduction bundle then
      add "  if (n <= 0) return LOOM_STATUS_INVALID_ARGUMENT;"
    else add "  if (n < 0) return LOOM_STATUS_INVALID_ARGUMENT;";
    List.iter
      (function
        | Tensor_ir.ScalarF32 _ -> ()
        | Tensor_ir.Tensor1F32 (name, _) ->
            add
              (Printf.sprintf
                 "  if (%s == nullptr) return LOOM_STATUS_INVALID_ARGUMENT;"
                 name))
      bundle.program.Tensor_ir.params;
    add "  if (out == nullptr) return LOOM_STATUS_INVALID_ARGUMENT;";
    add
      (Printf.sprintf "  size_t needed_workspace = %s(n);"
         bundle.workspace_symbol);
    add
      "  if (needed_workspace > 0 && (workspace == nullptr || workspace_size < \
       needed_workspace)) return LOOM_STATUS_WORKSPACE_TOO_SMALL;");
  if bundle_uses_workspace bundle then (
    add
      "  unsigned char* workspace_bytes = static_cast<unsigned \
       char*>(workspace);";
    add "  size_t workspace_offset = 0;";
    List.iter
      (function
        | Kernel_plan.Elementwise step when not (String.equal step.output "out") ->
            add "  workspace_offset = loom_align_up(workspace_offset, 256);";
            add
              (Printf.sprintf
                 "  float* %s = reinterpret_cast<float*>(workspace_bytes + \
                  workspace_offset);"
                 step.output);
            add "  workspace_offset += static_cast<size_t>(n) * sizeof(float);"
        | Kernel_plan.Reduction step ->
            if reduction_uses_double_atomic_workspace step then (
              add "  workspace_offset = loom_align_up(workspace_offset, 256);";
              add
                (Printf.sprintf
                   "  double* %s = reinterpret_cast<double*>(workspace_bytes + \
                    workspace_offset);"
                   (atomic_double_accumulator_name step));
              add "  workspace_offset += sizeof(double);")
            else if step.uses_workspace then
              let partials =
                Printf.sprintf "loom_choose_block_count_for(n, %d)"
                  (min_block_size step.block_size step.launch_buckets)
              in
              add "  workspace_offset = loom_align_up(workspace_offset, 256);";
              add
                (Printf.sprintf
                   "  float* %s_buf0 = reinterpret_cast<float*>(workspace_bytes \
                    + workspace_offset);"
                   step.kernel_name);
              add
                (Printf.sprintf
                   "  workspace_offset += static_cast<size_t>(%s) * \
                    sizeof(float);"
                   partials);
              add "  workspace_offset = loom_align_up(workspace_offset, 256);";
              add
                (Printf.sprintf
                   "  float* %s_buf1 = reinterpret_cast<float*>(workspace_bytes \
                    + workspace_offset);"
                   step.kernel_name);
              add
                (Printf.sprintf
                   "  workspace_offset += static_cast<size_t>(%s) * \
                    sizeof(float);"
                   partials)
            else if not (String.equal step.output "out") then (
              add "  workspace_offset = loom_align_up(workspace_offset, 256);";
              add
                (Printf.sprintf
                   "  float* %s = reinterpret_cast<float*>(workspace_bytes + \
                    workspace_offset);"
                   step.output);
              add "  workspace_offset += sizeof(float);")
        | Kernel_plan.Elementwise _ -> ())
      bundle.plan.Kernel_plan.steps);
  List.iter
    (function
      | Kernel_plan.Elementwise step ->
          let destination =
            if String.equal step.output "out" then "out" else step.output
          in
          let args =
            [ destination; "n" ] @ List.map snd step.inputs @ step.scalar_params
            |> String.concat ", "
          in
          add "  {";
          (match step.pointwise_family with
          | Kernel_plan.GenericPointwiseKernel ->
              if
                pointwise_prefers_medium_scalar_loop bundle.optimizations step
              then (
                add "    if (n <= 2097152) {";
                add
                  "      if (n <= static_cast<int64_t>(loom_kMaxBlocks) * \
                   loom_kBlockSize) {";
                add
                  (Printf.sprintf
                     "        %s_single<<<loom_choose_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "      } else {";
                add
                  (Printf.sprintf
                     "        %s_single<<<loom_choose_block_count_for(n, \
                      512), 512>>>(%s);"
                     step.kernel_name args);
                add "      }";
                add "    } else if (loom_pointwise_single_pass_supported_for(n, loom_kBlockSize)) {";
                add
                  (Printf.sprintf
                     "      %s_single<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    } else {";
                add
                  (Printf.sprintf
                     "      %s<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    }")
              else if
                pointwise_prefers_hft_base_block_cap bundle.optimizations step
              then (
                add "    if (n <= 2097152) {";
                add
                  "      if (n <= static_cast<int64_t>(loom_kMaxBlocks) * \
                   loom_kBlockSize) {";
                add
                  (Printf.sprintf
                     "        %s_single<<<loom_choose_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "      } else {";
                add
                  (Printf.sprintf
                     "        %s<<<loom_choose_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "      }";
                add
                  "    } else if (loom_pointwise_single_pass_supported_for(n, \
                   loom_kBlockSize)) {";
                add
                  (Printf.sprintf
                     "      %s_single<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    } else {";
                add
                  (Printf.sprintf
                     "      %s<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    }")
              else (
                add
                  "    if (loom_pointwise_single_pass_supported_for(n, \
                   loom_kBlockSize)) {";
                add
                  (Printf.sprintf
                     "      %s_single<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    } else {";
                add
                  (Printf.sprintf
                     "      %s<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                add "    }")
          | Kernel_plan.VectorizedPointwiseKernel ->
              let threshold_symbol =
                pointwise_vector_threshold_symbol bundle.optimizations
                  bundle.plan.Kernel_plan.entry_name step
              in
              let vector_single_step = vector_single_pass_step step in
              let add_vector_launch ?(single_pass = false) indent =
                let kernel_name =
                  if single_pass then vector_single_step.kernel_name
                  else step.kernel_name
                in
                add
                  (Printf.sprintf
                     "%sint64_t vector_n = loom_ceil_div_i64(n, 4);" indent);
                add
                  (Printf.sprintf
                     "%sint blocks = \
                      loom_choose_pointwise_block_count_for(vector_n, \
                      loom_kBlockSize);"
                     indent);
                add
                  (Printf.sprintf
                     "%s%s<<<blocks, loom_kBlockSize>>>(%s);" indent
                     kernel_name args)
              in
              let add_selected_vector_launch indent =
                if
                  pointwise_prefers_vector_single_pass bundle.optimizations step
                then (
                  add
                    (Printf.sprintf
                       "%sif (loom_pointwise_single_pass_supported_for(loom_ceil_div_i64(n, 4), loom_kBlockSize)) {"
                       indent);
                  add_vector_launch ~single_pass:true (indent ^ "  ");
                  add (Printf.sprintf "%s} else {" indent);
                  add_vector_launch (indent ^ "  ");
                  add (Printf.sprintf "%s}" indent))
                else add_vector_launch indent
              in
              if String.equal threshold_symbol "0" then
                match pointwise_vector_upper_bound bundle.optimizations step with
                | Some upper ->
                    add (Printf.sprintf "    if (n > %d) {" upper);
                    add
                      (Printf.sprintf
                         "      %s_scalar<<<loom_choose_pointwise_block_count(n), \
                          loom_kBlockSize>>>(%s);"
                         step.kernel_name args);
                    add "    } else {";
                    add_selected_vector_launch "      ";
                    add "    }"
                | None -> add_selected_vector_launch "    "
              else (
                add (Printf.sprintf "    if (n < %s) {" threshold_symbol);
                add
                  (Printf.sprintf
                     "      %s_scalar_single<<<loom_choose_pointwise_block_count(n), \
                      loom_kBlockSize>>>(%s);"
                     step.kernel_name args);
                (match pointwise_vector_upper_bound bundle.optimizations step with
                | Some upper ->
                    add (Printf.sprintf "    } else if (n > %d) {" upper);
                    add
                      (Printf.sprintf
                         "      %s_scalar<<<loom_choose_pointwise_block_count(n), \
                          loom_kBlockSize>>>(%s);"
                         step.kernel_name args)
                | None -> ());
                add "    } else {";
                add_selected_vector_launch "      ";
                add "    }")
          | Kernel_plan.SmallNPointwiseKernel ->
              List.iter add
                (pick_block_lines "    " "n" step.block_size
                   step.launch_buckets);
              add
                "    int blocks = loom_choose_pointwise_block_count_for(n, block_size);";
              add
                (Printf.sprintf "    %s<<<blocks, block_size>>>(%s);"
                   step.kernel_name args));
          add "    cudaError_t status = cudaGetLastError();";
          add "    if (status != cudaSuccess) return LOOM_STATUS_CUDA_ERROR;";
          add "  }"
      | Kernel_plan.Reduction step ->
          let destination =
            if String.equal step.output "out" then "out" else step.output
          in
          let atomic_destination =
            if reduction_uses_double_atomic_workspace step then
              atomic_double_accumulator_name step
            else destination
          in
          let first_args =
            match step.source with
            | Kernel_plan.PlainInput input ->
                (match step.execution_family with
                | Kernel_plan.AtomicOutputReduction ->
                    [ atomic_destination; input; "current_n" ]
                | _ -> [ input; "current"; "current_n" ])
            | Kernel_plan.MappedInput { inputs; scalar_params; _ } ->
                (match step.execution_family with
                | Kernel_plan.AtomicOutputReduction ->
                    [ atomic_destination; "current_n" ]
                | _ -> [ "current"; "current_n" ])
                @ List.map snd inputs @ scalar_params
          in
          add "  {";
          add "    int64_t current_n = n;";
          (match step.execution_family with
          | Kernel_plan.AtomicOutputReduction ->
              add
                (Printf.sprintf
                   "    cudaError_t status = cudaMemset(%s, 0, sizeof(%s));"
                   atomic_destination
                   (if reduction_uses_double_atomic_workspace step then
                      "double"
                    else "float"));
              add
                "    if (status != cudaSuccess) return \
                 LOOM_STATUS_CUDA_ERROR;";
              List.iter add
                (pick_block_lines "    " "current_n" step.block_size
                   step.launch_buckets);
              add
                "    int blocks = loom_choose_reduction_block_count_for(current_n, \
                 block_size);";
              add
                (Printf.sprintf "    %s<<<blocks, block_size>>>(%s);"
                   step.kernel_name
                   (String.concat ", " first_args));
              add "    status = cudaGetLastError();";
              add
                "    if (status != cudaSuccess) return LOOM_STATUS_CUDA_ERROR;"
              ;
              if reduction_uses_double_atomic_workspace step then (
                add
                  (Printf.sprintf
                     "    %s_finalize<<<1, 1>>>(%s, %s);" step.kernel_name
                     destination atomic_destination);
                add "    status = cudaGetLastError();";
                add
                  "    if (status != cudaSuccess) return \
                   LOOM_STATUS_CUDA_ERROR;")
          | _ ->
              let use_mid_stage_cap =
                reduction_uses_mid_stage_cap bundle.optimizations step
              in
              add (Printf.sprintf "    float* current = %s_buf0;" step.kernel_name);
              add (Printf.sprintf "    float* next = %s_buf1;" step.kernel_name);
              List.iter add
                (pick_block_lines "    " "current_n" step.block_size
                   step.launch_buckets);
              add
                "    int blocks = loom_choose_block_count_for(current_n, \
                 block_size);";
              if use_mid_stage_cap then
                add
                  "    if (current_n <= 2097152) blocks = \
                   loom_choose_reduction_block_count_for(current_n, \
                   block_size);";
              add
                (Printf.sprintf "    %s<<<blocks, block_size>>>(%s);"
                   step.kernel_name
                   (String.concat ", " first_args));
              add "    cudaError_t status = cudaGetLastError();";
              add
                "    if (status != cudaSuccess) return LOOM_STATUS_CUDA_ERROR;";
              add "    current_n = blocks;";
              (match step.single_block_threshold with
              | Some threshold ->
                  add (Printf.sprintf "    if (n <= %d) {" threshold);
                  add
                    (Printf.sprintf
                       "      status = cudaMemcpy(%s, current, sizeof(float), \
                        cudaMemcpyDeviceToDevice);"
                       destination);
                  add
                    "      if (status != cudaSuccess) return \
                     LOOM_STATUS_CUDA_ERROR;";
                  add "    } else {"
              | None -> ());
              add "    while (current_n > 1) {";
                  List.iter add
                    (pick_block_lines "      " "current_n" step.block_size
                       step.launch_buckets);
              add
                "      int current_blocks = \
                 loom_choose_block_count_for(current_n, block_size);";
              if use_mid_stage_cap then
                add
                  "      if (current_n <= 2097152) current_blocks = \
                   loom_choose_reduction_block_count_for(current_n, \
                   block_size);";
              add
                (Printf.sprintf
                   "      %s<<<current_blocks, block_size>>>(current, next, \
                    current_n);"
                   step.combine_kernel_name);
              add "      status = cudaGetLastError();";
              add
                "      if (status != cudaSuccess) return \
                 LOOM_STATUS_CUDA_ERROR;";
              add "      float* temp = current;";
              add "      current = next;";
              add "      next = temp;";
              add "      current_n = current_blocks;";
              add "    }";
              add
                (Printf.sprintf
                   "    status = cudaMemcpy(%s, current, sizeof(float), \
                    cudaMemcpyDeviceToDevice);"
                   destination);
              add "    if (status != cudaSuccess) return LOOM_STATUS_CUDA_ERROR;";
              (match step.single_block_threshold with
              | Some _ -> add "    }"
              | None -> ()));
          add "  }")
    bundle.plan.Kernel_plan.steps;
  List.iter add (render_copy_from_result bundle);
  if
    Optimizations.enabled bundle.optimizations
      Optimizations.CudaAsyncWrapperReturn
  then add "  return LOOM_STATUS_OK;"
  else (
    add "  cudaError_t final_status = cudaDeviceSynchronize();";
    add
      "  return final_status == cudaSuccess ? LOOM_STATUS_OK : \
       LOOM_STATUS_CUDA_ERROR;");
  add "}";
  add "";
  String.concat "\n" !lines

let entry_output_dir out_dir (bundle : entry_bundle) =
  Filename.concat (Filename.concat out_dir "entries") bundle.symbol_name

let write_entry_artifacts out_dir (bundle : entry_bundle) =
  let entry_dir = entry_output_dir out_dir bundle in
  let files =
    [
      ("lambda.sexp", Option.value bundle.lowered_sexp ~default:"; unavailable\n");
      ("loom_lambda.json", bundle.loom_entry_json);
      ("tensor_ir.json", bundle.tensor_ir_json);
      ("kernel_plan.json", bundle.kernel_plan_json);
      ("cuda_plan.json", bundle.cuda_plan_json);
      ("backend_analysis.json", bundle.backend_analysis_json);
      ( "report.md",
        String.concat "\n"
          [
            "# Loom Entry Report";
            "";
            Printf.sprintf "- entrypoint: `%s`" bundle.entry_name;
            Printf.sprintf "- module: `%s`" bundle.module_name;
            Printf.sprintf "- source: `%s`" bundle.source_file;
            "- pipeline: `FrontIR -> LoomLambda -> TensorIR -> KernelPlan traits -> CudaPlan -> CUDA`";
            Printf.sprintf "- exported symbol: `%s`" bundle.symbol_name;
            Printf.sprintf "- workspace function: `%s`" bundle.workspace_symbol;
            Printf.sprintf "- result kind: `%s`"
              (match bundle.program.Tensor_ir.result with
              | Tensor_ir.TensorResult _ -> "tensor"
              | Tensor_ir.ScalarResult _ -> "scalar");
           Printf.sprintf "- generated kernels: `%d`"
              (List.length bundle.plan.Kernel_plan.steps);
            Printf.sprintf "- temporary allocations: `%d`"
              bundle.plan.Kernel_plan.temporary_count;
            "";
          ] );
    ]
  in
  List.map
    (fun (name, body) ->
      let path = Filename.concat entry_dir name in
      write_file path body;
      Filename.concat (Filename.concat "entries" bundle.symbol_name) name)
    files

let render_header package_name (bundles : entry_bundle list) =
  let prototypes =
    bundles
    |> List.concat_map render_header_prototype
    |> List.map (fun line -> line ^ "\n")
    |> String.concat ""
  in
  String.concat ""
    [
      "#pragma once\n\n";
      "#include <stddef.h>\n";
      "#include <stdint.h>\n\n";
      "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n";
      "#define LOOM_STATUS_OK 0\n";
      "#define LOOM_STATUS_INVALID_ARGUMENT 1\n";
      "#define LOOM_STATUS_WORKSPACE_TOO_SMALL 2\n";
      "#define LOOM_STATUS_CUDA_ERROR 3\n\n";
      "/* Generated by Loom CUDA backend for ";
      package_name;
      ". */\n\n";
      prototypes;
      "\n#ifdef __cplusplus\n}\n#endif\n";
    ]

let cuda_tuning_constants = function
  | GenericPlatform -> (256, 4096, 32768, 262144)
  | CurrentPlatform -> (256, 4096, 32768, 262144)

let small_vector_pointwise_threshold = 262144
let render_source cuda_platform header_include (bundles : entry_bundle list) =
  let block_size, max_blocks, pointwise_max_blocks, vector_threshold =
    cuda_tuning_constants cuda_platform
  in
  let support =
    [
      "#include <cuda_runtime.h>";
      "#include <math.h>";
      "#include <stddef.h>";
      "#include <stdint.h>";
      Printf.sprintf "#include %S" header_include;
      "";
      Printf.sprintf "static constexpr int loom_kBlockSize = %d;" block_size;
      Printf.sprintf "static constexpr int loom_kMaxBlocks = %d;" max_blocks;
      Printf.sprintf "static constexpr int loom_kPointwiseMaxBlocks = %d;"
        pointwise_max_blocks;
      "static constexpr int loom_kMidReductionBlocks = 1024;";
      "static constexpr int loom_kMaxReductionBlocks = 2048;";
      Printf.sprintf
        "static constexpr int64_t loom_kVectorPointwiseThreshold = %d;"
        vector_threshold;
      Printf.sprintf
        "static constexpr int64_t loom_kSmallVectorPointwiseThreshold = %d;"
        small_vector_pointwise_threshold;
      "";
      "static inline int64_t loom_ceil_div_i64(int64_t n, int64_t d) {";
      "  return (n + d - 1) / d;";
      "}";
      "";
      "static inline size_t loom_align_up(size_t value, size_t alignment) {";
      "  return ((value + alignment - 1) / alignment) * alignment;";
      "}";
      "";
      "static inline int loom_choose_block_count_for(int64_t n, int \
       block_size) {";
      "  if (n <= 0) return 1;";
      "  if (block_size <= 0) block_size = loom_kBlockSize;";
      "  int64_t blocks = (n + block_size - 1) / block_size;";
      "  if (blocks < 1) blocks = 1;";
      "  if (blocks > loom_kMaxBlocks) blocks = loom_kMaxBlocks;";
      "  return static_cast<int>(blocks);";
      "}";
      "";
      "static inline int loom_choose_block_count(int64_t n) {";
      "  return loom_choose_block_count_for(n, loom_kBlockSize);";
      "}";
      "";
      "static inline int loom_choose_pointwise_block_count_for(int64_t n, int \
       block_size) {";
      "  if (n <= 0) return 1;";
      "  if (block_size <= 0) block_size = loom_kBlockSize;";
      "  int64_t blocks = (n + block_size - 1) / block_size;";
      "  if (blocks < 1) blocks = 1;";
      "  if (blocks > loom_kPointwiseMaxBlocks) blocks = \
       loom_kPointwiseMaxBlocks;";
      "  return static_cast<int>(blocks);";
      "}";
      "";
      "static inline int loom_choose_pointwise_block_count(int64_t n) {";
      "  return loom_choose_pointwise_block_count_for(n, loom_kBlockSize);";
      "}";
      "";
      "static inline bool loom_pointwise_single_pass_supported_for(int64_t n, \
       int block_size) {";
      "  if (block_size <= 0) block_size = loom_kBlockSize;";
      "  return n <= static_cast<int64_t>(loom_kPointwiseMaxBlocks) * \
       block_size;";
      "}";
      "";
      "static inline int loom_choose_reduction_block_count_for(int64_t n, int \
       block_size) {";
      "  int blocks = loom_choose_block_count_for(n, block_size);";
      "  if (n <= 2097152 && blocks > loom_kMidReductionBlocks) blocks = \
       loom_kMidReductionBlocks;";
      "  if (blocks > loom_kMaxReductionBlocks) blocks = \
       loom_kMaxReductionBlocks;";
      "  return blocks;";
      "}";
      "";
      "__device__ __forceinline__ float loom_warp_reduce_sum(float value) {";
      "  for (int offset = 16; offset > 0; offset >>= 1) {";
      "    value += __shfl_down_sync(0xffffffff, value, offset);";
      "  }";
      "  return value;";
      "}";
      "";
      "__device__ __forceinline__ float loom_warp_reduce_max(float value) {";
      "  for (int offset = 16; offset > 0; offset >>= 1) {";
      "    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));";
      "  }";
      "  return value;";
      "}";
      "";
      "__device__ __forceinline__ float loom_block_reduce_sum(float value) {";
      "  __shared__ float warp_sums[32];";
      "  int lane = threadIdx.x & 31;";
      "  int warp_id = threadIdx.x >> 5;";
      "  value = loom_warp_reduce_sum(value);";
      "  if (lane == 0) warp_sums[warp_id] = value;";
      "  __syncthreads();";
      "  int warp_count = (blockDim.x + 31) >> 5;";
      "  value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;";
      "  if (warp_id == 0) value = loom_warp_reduce_sum(value);";
      "  return value;";
      "}";
      "";
      "__device__ __forceinline__ float loom_block_reduce_max(float value) {";
      "  __shared__ float warp_maxes[32];";
      "  int lane = threadIdx.x & 31;";
      "  int warp_id = threadIdx.x >> 5;";
      "  value = loom_warp_reduce_max(value);";
      "  if (lane == 0) warp_maxes[warp_id] = value;";
      "  __syncthreads();";
      "  int warp_count = (blockDim.x + 31) >> 5;";
      "  value = threadIdx.x < warp_count ? warp_maxes[lane] : -INFINITY;";
      "  if (warp_id == 0) value = loom_warp_reduce_max(value);";
      "  return value;";
      "}";
      "";
    ]
  in
  let sections =
    [
      String.concat "\n" support;
      String.concat "\n" (List.concat_map render_step_kernels bundles);
      String.concat "\n" (List.map render_workspace_function bundles);
      String.concat "\n"
        (List.map render_wrapper bundles
        @ List.filter_map
            (fun bundle ->
              if bundle_uses_workspace bundle then None
              else Some (render_wrapper ~no_workspace:true bundle))
            bundles);
    ]
  in
  String.concat "\n" sections

let param_json = function
  | Tensor_ir.ScalarF32 name ->
      `Assoc [ ("name", `String name); ("kind", `String "scalar-f32") ]
  | Tensor_ir.Tensor1F32 (name, shape_symbol) ->
      `Assoc
        [
          ("name", `String name);
          ("kind", `String "tensor1-f32");
          ("shape_symbol", `String shape_symbol);
        ]

let entry_json (bundle : entry_bundle) generated_files =
  let fields =
    [
      ("entry_name", `String bundle.entry_name);
      ("module_name", `String bundle.module_name);
      ("source_file", `String bundle.source_file);
      ("symbol_name", `String bundle.symbol_name);
      ("workspace_symbol", `String bundle.workspace_symbol);
      ( "result_kind",
        `String
          (match bundle.program.Tensor_ir.result with
          | Tensor_ir.TensorResult _ -> "tensor"
          | Tensor_ir.ScalarResult _ -> "scalar") );
      ("params", `List (List.map param_json bundle.program.Tensor_ir.params));
      ("kernel_count", `Int (List.length bundle.plan.Kernel_plan.steps));
      ("temporary_count", `Int bundle.plan.Kernel_plan.temporary_count);
      ( "generated_files",
        `List (List.map (fun path -> `String path) generated_files) );
    ]
  in
  let fields =
    if bundle_uses_workspace bundle then fields
    else
      fields
      @ [
          ( "no_workspace_symbol",
            `String (no_workspace_symbol_name bundle.symbol_name) );
        ]
  in
  `Assoc fields

let render_manifest ~mode ~project_root ~package_name ~kind ~cuda_arch
    ~cuda_platform ~artifact_path ~header_path ~source_path ~entry_files
    ~build_commands ~optimizations =
  let block_size, max_blocks, pointwise_max_blocks, vector_threshold =
    cuda_tuning_constants cuda_platform
  in
  `Assoc
    [
      ("mode", `String mode);
      ("target_backend", `String "cuda");
      ("project_root", `String project_root);
      ("package_name", `String package_name);
      ("artifact_kind", `String (package_kind_to_string kind));
      ("cuda_arch", `String cuda_arch);
      ("cuda_platform", `String (cuda_platform_to_string cuda_platform));
      ( "cuda_tuning",
        `Assoc
          [
            ("block_size", `Int block_size);
            ("max_blocks", `Int max_blocks);
            ("pointwise_max_blocks", `Int pointwise_max_blocks);
            ("vector_pointwise_threshold", `Int vector_threshold);
          ] );
      ("artifact_path", `String artifact_path);
      ("header_path", `String header_path);
      ("source_path", `String source_path);
      ( "entries",
        `List
          (List.map
             (fun (bundle, files) -> entry_json bundle files)
             entry_files) );
      ("optimizations", Optimizations.to_yojson optimizations);
      ( "build_commands",
        `List (List.map (fun cmd -> `String cmd) build_commands) );
    ]
  |> Yojson.Safe.pretty_to_string

let render_report ~title ~project_root ~package_name ~kind ~cuda_arch
    ~cuda_platform ~artifact_path ~header_path ~source_path
    ~(bundles : entry_bundle list) ~build_commands
    ~optimizations =
  let block_size, max_blocks, pointwise_max_blocks, vector_threshold =
    cuda_tuning_constants cuda_platform
  in
  String.concat "\n"
    ([
       title;
       "";
       Printf.sprintf "- project root: `%s`" project_root;
       "- pipeline: `FrontIR -> LoomLambda -> TensorIR -> KernelPlan traits -> CudaPlan -> CUDA`";
       Printf.sprintf "- package: `%s`" package_name;
       Printf.sprintf "- artifact kind: `%s`" (package_kind_to_string kind);
       Printf.sprintf "- CUDA arch: `%s`" cuda_arch;
       Printf.sprintf "- CUDA platform: `%s`"
         (cuda_platform_to_string cuda_platform);
       Printf.sprintf
         "- CUDA tuning: block_size `%d`, max_blocks `%d`, \
          pointwise_max_blocks `%d`, vector_threshold `%d`"
         block_size max_blocks pointwise_max_blocks vector_threshold;
       Printf.sprintf "- artifact: `%s`" artifact_path;
       Printf.sprintf "- header: `%s`" header_path;
       Printf.sprintf "- source: `%s`" source_path;
       Printf.sprintf "- exported entrypoints: `%d`" (List.length bundles);
       Printf.sprintf "- enabled optimizations: %s"
         (match Optimizations.to_string_list optimizations with
         | [] -> "none"
         | names -> String.concat ", " names);
       "";
     ]
    @ List.concat_map
        (fun (bundle : entry_bundle) ->
          [
            Printf.sprintf "## `%s`" bundle.symbol_name;
            "";
            Printf.sprintf "- entrypoint: `%s`" bundle.entry_name;
            Printf.sprintf "- source: `%s`" bundle.source_file;
            Printf.sprintf "- result kind: `%s`"
              (match bundle.program.Tensor_ir.result with
              | Tensor_ir.TensorResult _ -> "tensor"
              | Tensor_ir.ScalarResult _ -> "scalar");
            Printf.sprintf "- generated kernels: `%d`"
              (List.length bundle.plan.Kernel_plan.steps);
            Printf.sprintf "- temporary allocations: `%d`"
              bundle.plan.Kernel_plan.temporary_count;
            "";
          ])
        bundles
    @ [ "## Build Commands"; "" ]
    @ List.map (fun command -> "- `" ^ command ^ "`") build_commands
    @ [ "" ])

let platform_nvcc_flags = function
  | GenericPlatform -> []
  | CurrentPlatform -> [ "-Xptxas"; "-O3" ]

let build_artifact ~kind ~cuda_arch ~cuda_platform ~include_dir ~source_path
    ~artifact_path =
  if not (command_available "nvcc") then
    Diagnostic.raise_error "nvcc is required on PATH for the Loom CUDA backend";
  let platform_flags = platform_nvcc_flags cuda_platform in
  match kind with
  | Shared ->
      let command =
        String.concat " "
          ([
             "nvcc";
             "-std=c++17";
             "-O3";
             "-shared";
             "-Xcompiler";
             "-fPIC";
             "-arch=" ^ cuda_arch;
           ]
          @ platform_flags
          @ [
              "-I" ^ quote include_dir;
              "-o";
              quote artifact_path;
              quote source_path;
            ])
      in
      run_command command;
      [ command ]
  | Static ->
      if not (command_available "ar") then
        Diagnostic.raise_error "ar is required on PATH for static CUDA artifacts";
      let object_path = Filename.chop_extension source_path ^ ".o" in
      let compile_command =
        String.concat " "
          ([
             "nvcc";
             "-std=c++17";
             "-O3";
             "-c";
             "-arch=" ^ cuda_arch;
           ]
          @ platform_flags
          @ [
              "-I" ^ quote include_dir;
              "-o";
              quote object_path;
              quote source_path;
            ])
      in
      let archive_command =
        String.concat " "
          [ "ar"; "rcs"; quote artifact_path; quote object_path ]
      in
      run_command compile_command;
      run_command archive_command;
      [ compile_command; archive_command ]

let compile_entry ~source_file ~module_name ~entry_name ~lowered_sexp
    ~loom_entry ~program ~plan ~out_dir ~cuda_arch ~cuda_platform
    ~optimizations =
  let bundle =
    make_bundle ~source_file ~module_name ~entry_name ~lowered_sexp ~loom_entry
      ~program ~plan ~cuda_platform ~optimizations
  in
  let package_name = bundle.symbol_name in
  let cuda_arch =
    match cuda_arch with Some value -> value | None -> detect_cuda_arch ()
  in
  ensure_dir out_dir;
  let header_path = Filename.concat out_dir (entry_name ^ "_cuda.h") in
  let source_path = Filename.concat out_dir (entry_name ^ "_cuda.cu") in
  let artifact_path = Filename.concat out_dir (Printf.sprintf "lib%s.so" entry_name) in
  let manifest_path = Filename.concat out_dir "manifest.json" in
  let report_path = Filename.concat out_dir "report.md" in
  let header_contents = render_header package_name [ bundle ] in
  let source_contents =
    render_source cuda_platform (Filename.basename header_path) [ bundle ]
  in
  write_file header_path header_contents;
  write_file source_path source_contents;
  let build_commands =
    build_artifact ~kind:Shared ~cuda_arch ~cuda_platform ~include_dir:out_dir
      ~source_path ~artifact_path
  in
  let generated_files =
    List.map Filename.basename [ header_path; source_path; artifact_path ]
  in
  let manifest_contents =
    render_manifest ~mode:"compile" ~project_root:(Filename.dirname source_file)
      ~package_name ~kind:Shared ~cuda_arch ~cuda_platform ~artifact_path
      ~header_path ~source_path ~entry_files:[ (bundle, generated_files) ]
      ~build_commands ~optimizations
  in
  let report_contents =
    render_report ~title:"# Loom CUDA Compile Report"
      ~project_root:(Filename.dirname source_file) ~package_name ~kind:Shared
      ~cuda_arch ~cuda_platform ~artifact_path ~header_path ~source_path
      ~bundles:[ bundle ] ~build_commands ~optimizations
  in
  write_file manifest_path manifest_contents;
  write_file report_path report_contents;
  {
    source_file;
    entry_name;
    module_name;
    symbol_name = bundle.symbol_name;
    workspace_symbol = bundle.workspace_symbol;
    artifact_path;
    header_path;
    source_path;
    manifest_path;
    report_path;
    generated_files;
  }

let package_project ~project_root ~out_dir ~kind ~input_kind ~module_filters
    ~entry_filters ~cuda_arch ~cuda_platform ~optimizations =
  let project_root =
    match input_kind with
    | OcamlFrontend -> find_project_root project_root
    | PythonFrontend -> find_python_project_root project_root
    | CppFrontend -> find_python_project_root project_root
    | AutoFrontend -> (
        try find_marker_root "loom-package.json" project_root
        with Diagnostic.Error _ -> (
          try find_project_root project_root
          with Diagnostic.Error _ -> find_python_project_root project_root ) )
  in
  let package_name =
    match input_kind with
    | PythonFrontend | CppFrontend -> package_name_of_python_root project_root
    | OcamlFrontend -> package_name_of_root project_root
    | AutoFrontend ->
        let marker = Filename.concat project_root "loom-package.json" in
        if Sys.file_exists marker then package_name_of_python_root project_root
        else package_name_of_root project_root
  in
  let bundles =
    discover_entries ~input_kind ~cuda_platform ~optimizations ~project_root
      ~module_filters ~entry_filters
  in
  if bundles = [] then
    Diagnostic.raise_error
      "no Loom entry bindings ([@loom.entry], @loom.entry, or LOOM_ENTRY) matched the requested project selection";
  let cuda_arch =
    match cuda_arch with Some value -> value | None -> detect_cuda_arch ()
  in
  let include_root = Filename.concat out_dir "include" in
  let header_dir = Filename.concat include_root "loom" in
  let source_dir = Filename.concat out_dir "src-gen" in
  ensure_dir header_dir;
  ensure_dir source_dir;
  let header_path = Filename.concat header_dir (package_name ^ ".h") in
  let source_path = Filename.concat source_dir (package_name ^ ".cu") in
  let artifact_path =
    Filename.concat out_dir
      (Printf.sprintf "lib%s.%s" package_name
         (match kind with Shared -> "so" | Static -> "a"))
  in
  let manifest_path = Filename.concat out_dir "manifest.json" in
  let report_path = Filename.concat out_dir "report.md" in
  let header_contents = render_header package_name bundles in
  let source_contents =
    render_source cuda_platform
      (Filename.concat "loom" (package_name ^ ".h"))
      bundles
  in
  write_file header_path header_contents;
  write_file source_path source_contents;
  let entry_files =
    List.map
      (fun bundle -> (bundle, write_entry_artifacts out_dir bundle))
      bundles
  in
  let build_commands =
    build_artifact ~kind ~cuda_arch ~cuda_platform ~include_dir:include_root
      ~source_path ~artifact_path
  in
  let manifest_contents =
    render_manifest ~mode:"package" ~project_root ~package_name ~kind ~cuda_arch
      ~cuda_platform ~artifact_path ~header_path ~source_path ~entry_files
      ~build_commands ~optimizations
  in
  let report_contents =
    render_report ~title:"# Loom Package Report" ~project_root ~package_name
      ~kind ~cuda_arch ~cuda_platform ~artifact_path ~header_path ~source_path
      ~bundles ~build_commands ~optimizations
  in
  write_file manifest_path manifest_contents;
  write_file report_path report_contents;
  {
    project_root;
    package_name;
    artifact_path;
    header_path;
    source_path;
    manifest_path;
    report_path;
  }
