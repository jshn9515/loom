type package_kind = Cuda_backend.package_kind =
  | Shared
  | Static

type cuda_platform = Cuda_backend.cuda_platform =
  | GenericPlatform
  | CurrentPlatform

type frontend_kind = Cuda_backend.frontend_kind =
  | OcamlFrontend
  | PythonFrontend
  | CppFrontend
  | AutoFrontend

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
