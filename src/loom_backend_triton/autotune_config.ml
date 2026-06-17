type config = {
  block_size : int;
  num_warps : int;
  num_stages : int;
}

type family = {
  configs : config list;
}

type t = {
  version : int;
  bucket_upper_bounds : int list;
  elementwise : family;
  reduction : family;
  source_path : string option;
}

let default_path = "configs/triton/default-v1.json"

let json_assoc fields = `Assoc fields

let expect_assoc = function
  | `Assoc fields -> fields
  | _ -> Diagnostic.raise_error "expected JSON object in Triton autotune config"

let expect_int field = function
  | `Int value -> value
  | _ -> Diagnostic.raise_error (Printf.sprintf "expected integer field %s in Triton autotune config" field)

let expect_list field = function
  | `List items -> items
  | _ -> Diagnostic.raise_error (Printf.sprintf "expected list field %s in Triton autotune config" field)

let field fields name =
  match List.assoc_opt name fields with
  | Some value -> value
  | None -> Diagnostic.raise_error (Printf.sprintf "missing Triton autotune config field %s" name)

let config_of_yojson json =
  let fields = expect_assoc json in
  { block_size = field fields "block_size" |> expect_int "block_size"
  ; num_warps = field fields "num_warps" |> expect_int "num_warps"
  ; num_stages = field fields "num_stages" |> expect_int "num_stages" }

let family_of_yojson json =
  let fields = expect_assoc json in
  let configs =
    field fields "configs" |> expect_list "configs" |> List.map config_of_yojson
  in
  if configs = [] then Diagnostic.raise_error "autotune config families must contain at least one config";
  { configs }

let load path =
  let json = Yojson.Safe.from_file path in
  let fields = expect_assoc json in
  { version = field fields "version" |> expect_int "version"
  ; bucket_upper_bounds =
      field fields "bucket_upper_bounds"
      |> expect_list "bucket_upper_bounds"
      |> List.map (expect_int "bucket_upper_bounds")
  ; elementwise = field fields "elementwise" |> family_of_yojson
  ; reduction = field fields "reduction" |> family_of_yojson
  ; source_path = Some path }

let config_to_yojson config =
  json_assoc
    [ ("block_size", `Int config.block_size)
    ; ("num_warps", `Int config.num_warps)
    ; ("num_stages", `Int config.num_stages) ]

let family_to_yojson family =
  json_assoc [ ("configs", `List (List.map config_to_yojson family.configs)) ]

let to_yojson config =
  json_assoc
    [ ("version", `Int config.version)
    ; ("bucket_upper_bounds", `List (List.map (fun value -> `Int value) config.bucket_upper_bounds))
    ; ("elementwise", family_to_yojson config.elementwise)
    ; ("reduction", family_to_yojson config.reduction)
    ; ( "source_path"
      , match config.source_path with Some path -> `String path | None -> `Null ) ]

let to_string config = Yojson.Safe.pretty_to_string (to_yojson config)
