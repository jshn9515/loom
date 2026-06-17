open Optimization_types

let default_config_path = "configs/optimizations/current.json"

let default_config =
  {
    version = 4;
    small_inline_nodes = 18;
    small_clone_nodes = 12;
    reduction_precombine_max_body_complexity = 18;
    single_block_reduction_threshold = 1024;
    reduction_stage_split_threshold = 65536;
    branchy_fusion_complexity_threshold = 6;
    pointwise_materialization_complexity_threshold = 5;
    materialize_multi_use_complexity_threshold = 4;
    reduction_stage_medium_threshold = 8192;
    reduction_stage_large_threshold = 262144;
    small_direct_reduction_threshold = 8192;
    small_partial_reduction_threshold = 131072;
    small_reduction_program_count = 8;
    branchy_small_reduction_threshold = 131072;
    branchy_small_reduction_program_count = 4;
    reduction_body_cse_min_occurrences = 2;
    reduction_materialize_complexity_threshold = 5;
    reduction_late_fusion_complexity_threshold = 8;
    elementwise_buckets =
      [
        { max_n = 131072; block_size = 256; num_warps = 4 };
        { max_n = 2097152; block_size = 512; num_warps = 8 };
        { max_n = max_int; block_size = 1024; num_warps = 8 };
      ];
    reduction_buckets =
      [
        { max_n = 131072; block_size = 128; num_warps = 4 };
        { max_n = 2097152; block_size = 256; num_warps = 4 };
        { max_n = max_int; block_size = 512; num_warps = 8 };
      ];
    source_path = None;
  }

let none = { enabled = Id_set.empty; config = default_config }

let full infos =
  {
    enabled =
      List.fold_left (fun acc item -> Id_set.add item.id acc) Id_set.empty infos;
    config = default_config;
  }

let enable config id = { config with enabled = Id_set.add id config.enabled }

let disable config id =
  { config with enabled = Id_set.remove id config.enabled }

let enabled config id = Id_set.mem id config.enabled

let expect_assoc = function
  | `Assoc fields -> fields
  | _ ->
      Diagnostic.raise_error "expected JSON object in Loom optimization config"

let expect_int field = function
  | `Int value -> value
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected integer field %s in Loom optimization config"
           field)

let expect_list field = function
  | `List items -> items
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected list field %s in Loom optimization config"
           field)

let field fields name =
  match List.assoc_opt name fields with
  | Some value -> value
  | None ->
      Diagnostic.raise_error
        (Printf.sprintf "missing Loom optimization config field %s" name)

let field_opt fields name = List.assoc_opt name fields

let launch_bucket_of_yojson json =
  let fields = expect_assoc json in
  {
    max_n = field fields "max_n" |> expect_int "max_n";
    block_size = field fields "block_size" |> expect_int "block_size";
    num_warps = field fields "num_warps" |> expect_int "num_warps";
  }

let config_of_yojson source_path json =
  let fields = expect_assoc json in
  {
    version = field fields "version" |> expect_int "version";
    small_inline_nodes =
      field fields "small_inline_nodes" |> expect_int "small_inline_nodes";
    small_clone_nodes =
      (match field_opt fields "small_clone_nodes" with
      | Some value -> expect_int "small_clone_nodes" value
      | None -> default_config.small_clone_nodes);
    reduction_precombine_max_body_complexity =
      field fields "reduction_precombine_max_body_complexity"
      |> expect_int "reduction_precombine_max_body_complexity";
    single_block_reduction_threshold =
      field fields "single_block_reduction_threshold"
      |> expect_int "single_block_reduction_threshold";
    reduction_stage_split_threshold =
      (match field_opt fields "reduction_stage_split_threshold" with
      | Some value -> expect_int "reduction_stage_split_threshold" value
      | None -> default_config.reduction_stage_split_threshold);
    branchy_fusion_complexity_threshold =
      (match field_opt fields "branchy_fusion_complexity_threshold" with
      | Some value -> expect_int "branchy_fusion_complexity_threshold" value
      | None -> default_config.branchy_fusion_complexity_threshold);
    pointwise_materialization_complexity_threshold =
      (match field_opt fields "pointwise_materialization_complexity_threshold" with
      | Some value ->
          expect_int "pointwise_materialization_complexity_threshold" value
      | None -> default_config.pointwise_materialization_complexity_threshold);
    materialize_multi_use_complexity_threshold =
      (match field_opt fields "materialize_multi_use_complexity_threshold" with
      | Some value ->
          expect_int "materialize_multi_use_complexity_threshold" value
      | None -> default_config.materialize_multi_use_complexity_threshold);
    reduction_stage_medium_threshold =
      (match field_opt fields "reduction_stage_medium_threshold" with
      | Some value -> expect_int "reduction_stage_medium_threshold" value
      | None -> default_config.reduction_stage_medium_threshold);
    reduction_stage_large_threshold =
      (match field_opt fields "reduction_stage_large_threshold" with
      | Some value -> expect_int "reduction_stage_large_threshold" value
      | None -> default_config.reduction_stage_large_threshold);
    small_direct_reduction_threshold =
      (match field_opt fields "small_direct_reduction_threshold" with
      | Some value -> expect_int "small_direct_reduction_threshold" value
      | None -> default_config.small_direct_reduction_threshold);
    small_partial_reduction_threshold =
      (match field_opt fields "small_partial_reduction_threshold" with
      | Some value -> expect_int "small_partial_reduction_threshold" value
      | None -> default_config.small_partial_reduction_threshold);
    small_reduction_program_count =
      (match field_opt fields "small_reduction_program_count" with
      | Some value -> expect_int "small_reduction_program_count" value
      | None -> default_config.small_reduction_program_count);
    branchy_small_reduction_threshold =
      (match field_opt fields "branchy_small_reduction_threshold" with
      | Some value -> expect_int "branchy_small_reduction_threshold" value
      | None -> default_config.branchy_small_reduction_threshold);
    branchy_small_reduction_program_count =
      (match field_opt fields "branchy_small_reduction_program_count" with
      | Some value -> expect_int "branchy_small_reduction_program_count" value
      | None -> default_config.branchy_small_reduction_program_count);
    reduction_body_cse_min_occurrences =
      (match field_opt fields "reduction_body_cse_min_occurrences" with
      | Some value -> expect_int "reduction_body_cse_min_occurrences" value
      | None -> default_config.reduction_body_cse_min_occurrences);
    reduction_materialize_complexity_threshold =
      (match field_opt fields "reduction_materialize_complexity_threshold" with
      | Some value -> expect_int "reduction_materialize_complexity_threshold" value
      | None -> default_config.reduction_materialize_complexity_threshold);
    reduction_late_fusion_complexity_threshold =
      (match field_opt fields "reduction_late_fusion_complexity_threshold" with
      | Some value ->
          expect_int "reduction_late_fusion_complexity_threshold" value
      | None -> default_config.reduction_late_fusion_complexity_threshold);
    elementwise_buckets =
      field fields "elementwise_buckets"
      |> expect_list "elementwise_buckets"
      |> List.map launch_bucket_of_yojson;
    reduction_buckets =
      field fields "reduction_buckets"
      |> expect_list "reduction_buckets"
      |> List.map launch_bucket_of_yojson;
    source_path;
  }

let load_config path =
  path |> Yojson.Safe.from_file |> config_of_yojson (Some path)

let of_cli_flags ~parse_flag ~config_path ~enable_flags ~disable_flags =
  let parse_list kind flags =
    List.map
      (fun item ->
        match parse_flag item with
        | Some id -> id
        | None ->
            Diagnostic.raise_error
              (Printf.sprintf "unknown optimization flag %s in --%s-opt" item
                 kind))
      flags
  in
  let config =
    match config_path with
    | Some path -> load_config path
    | None ->
        if Sys.file_exists default_config_path then
          load_config default_config_path
        else default_config
  in
  let base = { enabled = Id_set.empty; config } in
  let with_enabled =
    parse_list "enable" enable_flags |> List.fold_left enable base
  in
  parse_list "disable" disable_flags |> List.fold_left disable with_enabled

let to_string_list ~all_infos config =
  all_infos
  |> List.filter_map (fun item ->
         if enabled config item.id then Some item.flag else None)

let launch_bucket_to_yojson bucket =
  `Assoc
    [
      ("max_n", `Int bucket.max_n);
      ("block_size", `Int bucket.block_size);
      ("num_warps", `Int bucket.num_warps);
    ]

let config_to_yojson config =
  `Assoc
    [
      ("version", `Int config.version);
      ("small_inline_nodes", `Int config.small_inline_nodes);
      ("small_clone_nodes", `Int config.small_clone_nodes);
      ( "reduction_precombine_max_body_complexity",
        `Int config.reduction_precombine_max_body_complexity );
      ( "single_block_reduction_threshold",
        `Int config.single_block_reduction_threshold );
      ( "reduction_stage_split_threshold",
        `Int config.reduction_stage_split_threshold );
      ( "branchy_fusion_complexity_threshold",
        `Int config.branchy_fusion_complexity_threshold );
      ( "pointwise_materialization_complexity_threshold",
        `Int config.pointwise_materialization_complexity_threshold );
      ( "materialize_multi_use_complexity_threshold",
        `Int config.materialize_multi_use_complexity_threshold );
      ( "reduction_stage_medium_threshold",
        `Int config.reduction_stage_medium_threshold );
      ( "reduction_stage_large_threshold",
        `Int config.reduction_stage_large_threshold );
      ( "small_direct_reduction_threshold",
        `Int config.small_direct_reduction_threshold );
      ( "small_partial_reduction_threshold",
        `Int config.small_partial_reduction_threshold );
      ( "small_reduction_program_count",
        `Int config.small_reduction_program_count );
      ( "branchy_small_reduction_threshold",
        `Int config.branchy_small_reduction_threshold );
      ( "branchy_small_reduction_program_count",
        `Int config.branchy_small_reduction_program_count );
      ( "reduction_body_cse_min_occurrences",
        `Int config.reduction_body_cse_min_occurrences );
      ( "reduction_materialize_complexity_threshold",
        `Int config.reduction_materialize_complexity_threshold );
      ( "reduction_late_fusion_complexity_threshold",
        `Int config.reduction_late_fusion_complexity_threshold );
      ( "elementwise_buckets",
        `List (List.map launch_bucket_to_yojson config.elementwise_buckets) );
      ( "reduction_buckets",
        `List (List.map launch_bucket_to_yojson config.reduction_buckets) );
      ( "source_path",
        match config.source_path with
        | Some path -> `String path
        | None -> `Null );
    ]

let to_yojson ~all_infos config =
  `Assoc
    [
      ("enabled", `List (List.map (fun name -> `String name) (to_string_list ~all_infos config)));
      ("config", config_to_yojson config.config);
    ]
