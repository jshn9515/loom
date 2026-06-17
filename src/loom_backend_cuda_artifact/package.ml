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

type packaged_result = Cuda_backend.packaged_result = {
  project_root : string;
  package_name : string;
  artifact_path : string;
  header_path : string;
  source_path : string;
  manifest_path : string;
  report_path : string;
}

let package_kind_to_string = Cuda_backend.package_kind_to_string
let cuda_platform_to_string = Cuda_backend.cuda_platform_to_string
let frontend_kind_to_string = Cuda_backend.frontend_kind_to_string
let parse_cuda_platform = Cuda_backend.parse_cuda_platform
let find_project_root = Cuda_backend.find_project_root
let package_project = Cuda_backend.package_project
