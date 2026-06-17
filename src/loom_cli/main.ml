type emit_target =
  | EmitFrontIr
  | EmitLambda
  | EmitLoomLambda
  | EmitTensorIr
  | EmitKernelPlan
  | EmitTritonPlan
  | EmitCudaPlan
  | EmitBackendAnalysis
  | EmitTriton
  | EmitCuda
  | EmitReport
  | EmitAll

type compile_target =
  | TargetTriton
  | TargetCuda

type input_kind = InputOcaml | InputPython | InputCpp | InputFrontIr

let usage () =
  prerr_endline "usage:";
  prerr_endline "  loomc list-entries <file> [--input-kind ocaml|python|cpp]";
  prerr_endline "  loomc front-ir <file> --entry <name> [--input-kind ocaml|python|cpp]";
  prerr_endline "  loomc list-opts";
  prerr_endline
    "  loomc compile <file> --entry <name> --target triton|cuda --out <dir> \
     [--emit <kind>] [--input-kind ocaml|python|cpp|front-ir] [--autotune] \
     [--autotune-config <path>] [--cuda-arch sm_XX] \
     [--cuda-platform generic|current] [--opt-config <path>] \
     [--enable-opt <id>] [--disable-opt <id>]";
  prerr_endline
    "  loomc package --project <dir> --out <dir> [--kind shared|static] \
     [--input-kind ocaml|python|cpp|auto] [--module <name>] [--entry <name>] [--cuda-arch sm_XX] \
     [--cuda-platform generic|current] [--opt-config <path>] \
     [--enable-opt <id>] [--disable-opt <id>]";
  exit 1

let ensure_dir path =
  let rec loop current =
    if current = "" || Sys.file_exists current then ()
    else (
      loop (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  loop path

let write_file path contents =
  let oc = open_out path in
  output_string oc contents;
  close_out oc

let parse_emit value =
  match value with
  | "front-ir" -> [ EmitFrontIr ]
  | "lambda" -> [ EmitLambda ]
  | "loom-lambda" -> [ EmitLoomLambda ]
  | "tensor-ir" -> [ EmitTensorIr ]
  | "kernel-plan" -> [ EmitKernelPlan ]
  | "triton-plan" -> [ EmitTritonPlan ]
  | "cuda-plan" -> [ EmitCudaPlan ]
  | "backend-analysis" -> [ EmitBackendAnalysis ]
  | "triton" -> [ EmitTriton ]
  | "cuda" -> [ EmitCuda ]
  | "report" -> [ EmitReport ]
  | "all" -> [ EmitAll ]
  | _ -> Diagnostic.raise_error (Printf.sprintf "unknown emit target %s" value)

let parse_compile_target value =
  match String.lowercase_ascii value with
  | "triton" -> TargetTriton
  | "cuda" -> TargetCuda
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "unknown compile target %s (expected triton or cuda)"
           value)

let validate_emit_for_target target emit =
  let emit = if List.mem EmitAll emit then [ EmitAll ] else emit in
  let incompatible =
    match target with
    | TargetTriton ->
        List.filter
          (function EmitCudaPlan | EmitCuda -> true | _ -> false)
          emit
    | TargetCuda ->
        List.filter
          (function EmitTritonPlan | EmitTriton -> true | _ -> false)
          emit
  in
  if incompatible <> [] then
    let render = function
      | EmitFrontIr -> "front-ir"
      | EmitLambda -> "lambda"
      | EmitLoomLambda -> "loom-lambda"
      | EmitTensorIr -> "tensor-ir"
      | EmitKernelPlan -> "kernel-plan"
      | EmitTritonPlan -> "triton-plan"
      | EmitCudaPlan -> "cuda-plan"
      | EmitBackendAnalysis -> "backend-analysis"
      | EmitTriton -> "triton"
      | EmitCuda -> "cuda"
      | EmitReport -> "report"
      | EmitAll -> "all"
    in
    let target_name =
      match target with TargetTriton -> "triton" | TargetCuda -> "cuda"
    in
    Diagnostic.raise_error
      (Printf.sprintf "--emit %s cannot be used with --target %s"
         (String.concat ", " (List.map render incompatible))
         target_name)

let is_cuda_specific_flag flag =
  let prefix = "cuda-" in
  String.length flag >= String.length prefix
  && String.sub flag 0 (String.length prefix) = prefix

let validate_optimizations_for_target target optimizations =
  match target with
  | TargetCuda -> ()
  | TargetTriton ->
      let cuda_flags =
        optimizations |> Optimizations.to_string_list
        |> List.filter is_cuda_specific_flag
      in
      if cuda_flags <> [] then
        Diagnostic.raise_error
          (Printf.sprintf
             "CUDA-specific optimization flags cannot be used with --target \
              triton: %s"
             (String.concat ", " cuda_flags))

let parse_package_kind value =
  match String.lowercase_ascii value with
  | "shared" | "dynamic" -> Package.Shared
  | "static" -> Package.Static
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "unknown package kind %s (expected shared or static)"
           value)

let parse_input_kind value =
  match String.lowercase_ascii value with
  | "ocaml" -> InputOcaml
  | "python" | "py" -> InputPython
  | "cpp" | "c++" -> InputCpp
  | "front-ir" | "front_ir" -> InputFrontIr
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "unknown input kind %s (expected ocaml, python, cpp, or front-ir)"
           value)

let parse_package_input_kind value =
  match String.lowercase_ascii value with
  | "ocaml" -> Cuda_backend.OcamlFrontend
  | "python" | "py" -> Cuda_backend.PythonFrontend
  | "cpp" | "c++" -> Cuda_backend.CppFrontend
  | "auto" -> Cuda_backend.AutoFrontend
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf
           "unknown package input kind %s (expected ocaml, python, cpp, or auto)"
           value)

let list_entries input_kind file =
  let entries =
    match input_kind with
    | InputOcaml ->
        Ocaml_entry_scan.list_entries file
        |> List.map (fun entry ->
               ( entry.Ocaml_entry_scan.name,
                 entry.params
                 |> List.map (fun (param : Ocaml_entry_scan.param_summary) ->
                        (param.name, param.kind)) ))
    | InputPython ->
        Python_frontend.list_entries file
        |> List.map (fun entry ->
               ( entry.Python_frontend.name,
                 entry.params
                 |> List.map (fun (param : Python_frontend.param_summary) ->
                        (param.name, param.kind)) ))
    | InputCpp ->
        Cpp_frontend.list_entries file
        |> List.map (fun entry ->
               ( entry.Cpp_frontend.name,
                 entry.params
                 |> List.map (fun (param : Cpp_frontend.param_summary) ->
                        (param.name, param.kind)) ))
    | InputFrontIr ->
        Diagnostic.raise_error "list-entries does not support --input-kind front-ir"
  in
  entries
  |> List.iter (fun (name, params) ->
      let rendered_params =
        params
        |> List.map (fun (name, kind) ->
            Printf.sprintf "%s:%s" name
              (Loom_types.entry_param_kind_to_string kind))
        |> String.concat ", "
      in
      print_endline (Printf.sprintf "%s (%s)" name rendered_params))

let load_frontend ~input_kind file entry_name =
  match input_kind with
  | InputOcaml ->
      let entry = Ocaml_entry_scan.find_entry file entry_name in
      let lowered = Some (Ocaml_raw_lambda.lower_entry entry) in
      let front_entry = Ocaml_front_ir.import_entry entry in
      (lowered, front_entry)
  | InputPython ->
      let front_entry = Python_frontend.import_entry file entry_name in
      (None, front_entry)
  | InputCpp ->
      let front_entry = Cpp_frontend.import_entry file entry_name in
      (None, front_entry)
  | InputFrontIr ->
      let front_entry =
        Front_ir.entry_of_string
          (Stdlib.In_channel.with_open_text file Stdlib.In_channel.input_all)
      in
      if not (String.equal front_entry.Front_ir.name entry_name) then
        Diagnostic.raise_error
          (Printf.sprintf "FrontIR entry %s does not match requested --entry %s"
             front_entry.name entry_name);
      (None, front_entry)

let compile file entry_name out_dir emit input_kind target autotune cuda_arch
    cuda_platform optimizations =
  let lowered, front_entry = load_frontend ~input_kind file entry_name in
  let lambda_entry = Normalize.entry_of_front_ir ~optimizations front_entry in
  let program = Tensorize.program_of_entry ~optimizations lambda_entry in
  let kernel_plan = Kernel_plan.of_program ~optimizations program in
  ensure_dir out_dir;
  let written = ref [] in
  let write name body =
    let path = Filename.concat out_dir name in
    write_file path body;
    written := name :: !written
  in
  let emit =
    if List.mem EmitAll emit then
      match target with
      | TargetTriton ->
          [
            EmitFrontIr;
            EmitLambda;
            EmitLoomLambda;
            EmitTensorIr;
            EmitKernelPlan;
            EmitTritonPlan;
            EmitBackendAnalysis;
            EmitTriton;
            EmitReport;
          ]
      | TargetCuda ->
          [
            EmitFrontIr;
            EmitLambda;
            EmitLoomLambda;
            EmitTensorIr;
            EmitKernelPlan;
            EmitCudaPlan;
            EmitBackendAnalysis;
            EmitCuda;
            EmitReport;
          ]
    else emit
  in
  if List.mem EmitFrontIr emit then
    write "front_ir.json" (Front_ir.entry_to_string front_entry);
  if List.mem EmitLambda emit then
    begin match lowered with
    | Some lowered ->
        write "lambda.sexp" (Ocaml_raw_lambda.raw_lambda_to_string lowered)
    | None -> write "lambda.sexp" "; unavailable for this frontend\n"
    end;
  if List.mem EmitLoomLambda emit then
    write "loom_lambda.json" (LoomLambda.entry_to_string lambda_entry);
  if List.mem EmitTensorIr emit then
    write "tensor_ir.json" (Tensor_ir.program_to_string program);
  if List.mem EmitKernelPlan emit then
    write "kernel_plan.json" (Kernel_plan.to_string kernel_plan);
  match target with
  | TargetTriton ->
      let triton_plan = Triton_plan.of_program ~optimizations program in
      if List.mem EmitTritonPlan emit then
        write "triton_plan.json" (Triton_plan.to_string triton_plan);
      if List.mem EmitBackendAnalysis emit then
        write "backend_analysis.json"
          (Triton_analysis.to_string ~program ~plan:triton_plan);
      write "pipeline.json"
        (Manifest.render_front_ir_manifest ~front_entry ~lambda_entry ~program
           ~kernel_plan ~backend_plan:triton_plan ~optimizations);
      let generated = Codegen.generate ~autotune ~program ~plan:triton_plan in
      if List.mem EmitTriton emit then write generated.filename generated.source;
      if List.mem EmitReport emit then
        write "report.md"
          (Manifest.render_report ~front_entry ~lambda_entry ~program
             ~plan:triton_plan
             ~autotune ~optimizations);
      write "manifest.json"
        (Manifest.render_manifest ~program ~plan:triton_plan ~autotune
           ~generated_files:(List.rev !written) ~optimizations)
  | TargetCuda ->
      let cuda_plan = Cuda_plan.of_program ~optimizations program in
      if List.mem EmitCudaPlan emit then
        write "cuda_plan.json" (Cuda_plan.to_string cuda_plan);
      if List.mem EmitBackendAnalysis emit then
        write "backend_analysis.json"
          (Cuda_analysis.to_string ~program ~plan:cuda_plan);
      write "pipeline.json"
        (Yojson.Safe.pretty_to_string
           (`Assoc
             [
               ("front_ir", Front_ir.entry_to_yojson front_entry);
               ("loom_lambda", LoomLambda.entry_to_yojson lambda_entry);
               ("tensor_ir", Tensor_ir.program_to_yojson program);
               ("kernel_plan", Kernel_plan.to_yojson kernel_plan);
               ("cuda_plan", Cuda_plan.to_yojson cuda_plan);
               ("optimizations", Optimizations.to_yojson optimizations);
             ]));
      let module_name = Cuda_backend.module_name_of_file file in
      let lowered_sexp =
        Option.map Ocaml_raw_lambda.raw_lambda_to_string lowered
      in
      ignore
        (Cuda_backend.compile_entry ~source_file:file ~module_name ~entry_name
           ~lowered_sexp ~loom_entry:lambda_entry ~program ~plan:cuda_plan
           ~out_dir ~cuda_arch ~cuda_platform ~optimizations)

let emit_front_ir file entry_name input_kind =
  match input_kind with
  | InputFrontIr ->
      Diagnostic.raise_error "front-ir command does not support --input-kind front-ir"
  | InputOcaml | InputPython | InputCpp ->
      let _, front_entry = load_frontend ~input_kind file entry_name in
      print_endline (Front_ir.entry_to_string front_entry)

let package_project project_root out_dir kind input_kind module_filters entry_filters
    cuda_arch cuda_platform optimizations =
  let result =
    Package.package_project ~project_root ~out_dir ~kind ~input_kind ~module_filters
      ~entry_filters ~cuda_arch ~cuda_platform ~optimizations
  in
  print_endline result.artifact_path

let run argv =
  match Array.to_list argv with
  | _ :: "list-entries" :: file :: rest ->
      let rec parse input_kind = function
        | [] ->
            list_entries input_kind file;
            0
        | "--input-kind" :: value :: tl -> parse (parse_input_kind value) tl
        | _ -> usage ()
      in
      parse InputOcaml rest
  | _ :: "front-ir" :: file :: rest ->
      let rec parse entry input_kind = function
        | [] ->
            let entry_name =
              match entry with Some value -> value | None -> usage ()
            in
            emit_front_ir file entry_name input_kind;
            0
        | "--entry" :: value :: tl -> parse (Some value) input_kind tl
        | "--input-kind" :: value :: tl -> parse entry (parse_input_kind value) tl
        | _ -> usage ()
      in
      parse None InputOcaml rest
  | [ _; "list-opts" ] ->
      print_string (Optimizations.render_cli_list ());
      0
  | _ :: "compile" :: file :: rest ->
      let rec parse entry target out emit input_kind autotune cuda_arch
          cuda_platform opt_config enable_flags disable_flags = function
        | [] ->
            let entry_name =
              match entry with Some value -> value | None -> usage ()
            in
            let target = match target with Some value -> value | None -> usage () in
            let out_dir =
              match out with Some value -> value | None -> usage ()
            in
            let emit =
              match emit with Some value -> parse_emit value | None -> [ EmitAll ]
            in
            let optimizations =
              Optimizations.of_cli_flags ~config_path:opt_config
                ~enable_flags:(List.rev enable_flags)
                ~disable_flags:(List.rev disable_flags)
            in
            let target = parse_compile_target target in
            validate_optimizations_for_target target optimizations;
            validate_emit_for_target target emit;
            (match (target, cuda_platform) with
            | TargetTriton, Some _ ->
                Diagnostic.raise_error
                  "--cuda-platform can only be used with --target cuda"
            | _ -> ());
            (match (target, cuda_arch) with
            | TargetTriton, Some _ ->
                Diagnostic.raise_error
                  "--cuda-arch can only be used with --target cuda"
            | _ -> ());
            (match target with
            | TargetTriton -> ()
            | TargetCuda ->
                if Option.is_some autotune then
                  Diagnostic.raise_error
                    "The generated CUDA backend does not support Triton autotuning flags");
            compile file entry_name out_dir emit input_kind target autotune cuda_arch
              (Option.value cuda_platform ~default:Package.GenericPlatform)
              optimizations;
            0
        | "--entry" :: value :: tl ->
            parse (Some value) target out emit input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--target" :: value :: tl ->
            parse entry (Some value) out emit input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--backend" :: value :: tl ->
            parse entry (Some value) out emit input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--out" :: value :: tl ->
            parse entry target (Some value) emit input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--emit" :: value :: tl ->
            parse entry target out (Some value) input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--input-kind" :: value :: tl ->
            parse entry target out emit (parse_input_kind value) autotune
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--autotune" :: tl ->
            parse entry target out emit input_kind
              (Some (Autotune_config.load Autotune_config.default_path))
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--autotune-config" :: value :: tl ->
            parse entry target out emit input_kind
              (Some (Autotune_config.load value))
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--cuda-arch" :: value :: tl ->
            parse entry target out emit input_kind autotune (Some value)
              cuda_platform opt_config enable_flags disable_flags tl
        | "--cuda-platform" :: value :: tl ->
            parse entry target out emit input_kind autotune cuda_arch
              (Some (Package.parse_cuda_platform value))
              opt_config enable_flags disable_flags tl
        | "--opt-config" :: value :: tl ->
            parse entry target out emit input_kind autotune cuda_arch
              cuda_platform (Some value) enable_flags disable_flags tl
        | "--enable-opt" :: value :: tl ->
            parse entry target out emit input_kind autotune cuda_arch
              cuda_platform opt_config (value :: enable_flags) disable_flags tl
        | "--disable-opt" :: value :: tl ->
            parse entry target out emit input_kind autotune cuda_arch
              cuda_platform opt_config enable_flags (value :: disable_flags) tl
        | _ -> usage ()
      in
      parse None None None None InputOcaml None None None None [] [] rest
  | _ :: "package" :: rest ->
      let rec parse project out kind input_kind module_filters entry_filters cuda_arch
          cuda_platform opt_config enable_flags disable_flags = function
        | [] ->
            let project_root =
              match project with Some value -> value | None -> usage ()
            in
            let out_dir =
              match out with Some value -> value | None -> usage ()
            in
            let kind =
              match kind with Some value -> value | None -> Package.Shared
            in
            let optimizations =
              Optimizations.of_cli_flags ~config_path:opt_config
                ~enable_flags:(List.rev enable_flags)
                ~disable_flags:(List.rev disable_flags)
            in
            package_project project_root out_dir kind input_kind (List.rev module_filters)
              (List.rev entry_filters) cuda_arch
              (Option.value cuda_platform ~default:Package.GenericPlatform)
              optimizations;
            0
        | "--project" :: value :: tl ->
            parse (Some value) out kind input_kind module_filters entry_filters cuda_arch
              cuda_platform opt_config enable_flags disable_flags tl
        | "--out" :: value :: tl ->
            parse project (Some value) kind input_kind module_filters entry_filters
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--kind" :: value :: tl ->
            parse project out
              (Some (parse_package_kind value))
              input_kind module_filters entry_filters cuda_arch cuda_platform opt_config
              enable_flags disable_flags tl
        | "--input-kind" :: value :: tl ->
            parse project out kind (parse_package_input_kind value)
              module_filters entry_filters cuda_arch cuda_platform opt_config
              enable_flags disable_flags tl
        | "--module" :: value :: tl ->
            parse project out kind input_kind (value :: module_filters) entry_filters
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--entry" :: value :: tl ->
            parse project out kind input_kind module_filters (value :: entry_filters)
              cuda_arch cuda_platform opt_config enable_flags disable_flags tl
        | "--cuda-arch" :: value :: tl ->
            parse project out kind input_kind module_filters entry_filters (Some value)
              cuda_platform opt_config enable_flags disable_flags tl
        | "--cuda-platform" :: value :: tl ->
            parse project out kind input_kind module_filters entry_filters cuda_arch
              (Some (Package.parse_cuda_platform value))
              opt_config enable_flags disable_flags tl
        | "--opt-config" :: value :: tl ->
            parse project out kind input_kind module_filters entry_filters cuda_arch
              cuda_platform (Some value) enable_flags disable_flags tl
        | "--enable-opt" :: value :: tl ->
            parse project out kind input_kind module_filters entry_filters cuda_arch
              cuda_platform opt_config (value :: enable_flags) disable_flags tl
        | "--disable-opt" :: value :: tl ->
            parse project out kind input_kind module_filters entry_filters cuda_arch
              cuda_platform opt_config enable_flags (value :: disable_flags) tl
        | _ -> usage ()
      in
      parse None None None Cuda_backend.OcamlFrontend [] [] None None None [] [] rest
  | _ -> usage ()

let () =
  match Diagnostic.protect (fun () -> run Sys.argv) with
  | Ok code -> exit code
  | Error diag ->
      prerr_endline (Diagnostic.to_string diag);
      exit 1
