type param_summary = {
  name : string;
  kind : Loom_types.entry_param_kind;
}

type entry_summary = {
  name : string;
  params : param_summary list;
}

let script_path () =
  match Sys.getenv_opt "LOOM_PYTHON_FRONTEND" with
  | Some path -> path
  | None -> Filename.concat (Sys.getcwd ()) "src/frontends/python/loom_frontend_python.py"

let python_executable () =
  match Sys.getenv_opt "PYTHON" with Some value -> value | None -> "python3"

let run_frontend args =
  let python = python_executable () in
  let argv = Array.of_list (python :: script_path () :: args) in
  let ic = Unix.open_process_args_in python argv in
  let output = Stdlib.In_channel.input_all ic in
  match Unix.close_process_in ic with
  | Unix.WEXITED 0 -> output
  | Unix.WEXITED code ->
      Diagnostic.raise_error
        (Printf.sprintf "Python frontend failed with exit status %d" code)
  | Unix.WSIGNALED signal ->
      Diagnostic.raise_error
        (Printf.sprintf "Python frontend was killed by signal %d" signal)
  | Unix.WSTOPPED signal ->
      Diagnostic.raise_error
        (Printf.sprintf "Python frontend was stopped by signal %d" signal)

let entry_param_kind_of_string = function
  | "scalar-f32" -> Loom_types.ScalarF32
  | "tensor1-f32" -> Loom_types.Tensor1F32
  | value ->
      Diagnostic.raise_error
        (Printf.sprintf "unknown Python frontend entry parameter kind %s" value)

let field fields name =
  match List.assoc_opt name fields with
  | Some value -> value
  | None ->
      Diagnostic.raise_error
        (Printf.sprintf "missing Python frontend JSON field %s" name)

let expect_assoc = function
  | `Assoc fields -> fields
  | _ -> Diagnostic.raise_error "expected Python frontend JSON object"

let expect_string field = function
  | `String value -> value
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected string for Python frontend field %s" field)

let expect_list field = function
  | `List values -> values
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected list for Python frontend field %s" field)

let param_summary_of_yojson json =
  let fields = expect_assoc json in
  {
    name = field fields "name" |> expect_string "name";
    kind =
      field fields "kind" |> expect_string "kind"
      |> entry_param_kind_of_string;
  }

let entry_summary_of_yojson json =
  let fields = expect_assoc json in
  {
    name = field fields "name" |> expect_string "name";
    params =
      field fields "params" |> expect_list "params"
      |> List.map param_summary_of_yojson;
  }

let list_entries file =
  let output = run_frontend [ "list-entries-json"; file ] in
  let fields = Yojson.Safe.from_string output |> expect_assoc in
  field fields "entries" |> expect_list "entries"
  |> List.map entry_summary_of_yojson

let import_entry file entry_name =
  let output = run_frontend [ "front-ir-json"; file; "--entry"; entry_name ] in
  Front_ir.entry_of_string output
