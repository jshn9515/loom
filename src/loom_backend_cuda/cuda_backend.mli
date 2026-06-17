type package_kind =
  | Shared
  | Static

type cuda_platform =
  | GenericPlatform
  | CurrentPlatform

type frontend_kind =
  | OcamlFrontend
  | PythonFrontend
  | CppFrontend
  | AutoFrontend

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

val package_kind_to_string : package_kind -> string
val cuda_platform_to_string : cuda_platform -> string
val frontend_kind_to_string : frontend_kind -> string
val parse_cuda_platform : string -> cuda_platform
val find_project_root : string -> string
val module_name_of_file : string -> string

val compile_entry :
  source_file:string ->
  module_name:string ->
  entry_name:string ->
  lowered_sexp:string option ->
  loom_entry:LoomLambda.entry ->
  program:Tensor_ir.program ->
  plan:Cuda_plan.t ->
  out_dir:string ->
  cuda_arch:string option ->
  cuda_platform:cuda_platform ->
  optimizations:Optimizations.t ->
  compile_result

val package_project :
  project_root:string ->
  out_dir:string ->
  kind:package_kind ->
  input_kind:frontend_kind ->
  module_filters:string list ->
  entry_filters:string list ->
  cuda_arch:string option ->
  cuda_platform:cuda_platform ->
  optimizations:Optimizations.t ->
  packaged_result
