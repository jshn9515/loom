open Typedtree

type primitive_identities = {
  map_path : Path.t;
  map2_path : Path.t;
  reduce_sum_path : Path.t;
  reduce_max_path : Path.t;
  tensor_type_path : Path.t;
}

type compilation_unit = {
  file : string;
  structure : Typedtree.structure;
  user_structure : Typedtree.structure;
  primitive_ids : primitive_identities;
}

type param_summary = {
  name : string;
  kind : Loom_types.entry_param_kind;
}

type entry = {
  owner : compilation_unit;
  name : string;
  loc : Location.t;
  binding : Typedtree.value_binding;
  params : param_summary list;
}

let prelude_source =
  {|
module Loom = struct
  module Tensor1 = struct
    type t

    let map (_ : float -> float) (_ : t) : t =
      invalid_arg "staged primitive: compile with loomc"

    let map2 (_ : float -> float -> float) (_ : t) (_ : t) : t =
      invalid_arg "staged primitive: compile with loomc"

    let reduce_sum (_ : t) : float =
      invalid_arg "staged primitive: compile with loomc"

    let reduce_max (_ : t) : float =
      invalid_arg "staged primitive: compile with loomc"
  end
end
|}

let longident3 a b c =
  match Longident.unflatten [ a; b; c ] with
  | Some lid -> lid
  | None -> invalid_arg "invalid longident"

let parse_string ~filename source =
  let lexbuf = Lexing.from_string source in
  Location.init lexbuf filename;
  Parse.implementation lexbuf

let rec drop n xs =
  if n <= 0 then xs
  else
    match xs with
    | [] -> []
    | _ :: rest -> drop (n - 1) rest

let load_file file =
  let prelude = parse_string ~filename:"<loom-prelude>" prelude_source in
  let prelude_count = List.length prelude in
  Compmisc.init_path ();
  let user = Pparse.parse_implementation ~tool_name:"loomc" file in
  let combined = prelude @ user in
  let typed_structure, _, _, _, _ =
    Typemod.type_structure (Compmisc.initial_env ()) combined
  in
  let user_items = drop prelude_count typed_structure.str_items in
  let final_env = typed_structure.str_final_env in
  let loc = Location.none in
  let map_path, _ = Env.lookup_value ~use:false ~loc (longident3 "Loom" "Tensor1" "map") final_env in
  let map2_path, _ = Env.lookup_value ~use:false ~loc (longident3 "Loom" "Tensor1" "map2") final_env in
  let reduce_sum_path, _ =
    Env.lookup_value ~use:false ~loc (longident3 "Loom" "Tensor1" "reduce_sum") final_env
  in
  let reduce_max_path, _ =
    Env.lookup_value ~use:false ~loc (longident3 "Loom" "Tensor1" "reduce_max") final_env
  in
  let tensor_type_path, _ =
    Env.lookup_type ~use:false ~loc (longident3 "Loom" "Tensor1" "t") final_env
  in
  { file
  ; structure = typed_structure
  ; user_structure =
      { typed_structure with Typedtree.str_items = user_items; str_final_env = final_env }
  ; primitive_ids =
      { map_path; map2_path; reduce_sum_path; reduce_max_path; tensor_type_path } }

let rec pattern_name (pat : Typedtree.pattern) =
  match pat.pat_desc with
  | Tpat_any -> "_"
  | Tpat_var (ident, _, _) -> Ident.name ident
  | Tpat_alias (inner, ident, _, _, _) ->
      let inner_name = pattern_name inner in
      if inner_name = "_" then Ident.name ident else inner_name
  | _ -> "_"

let has_loom_entry attrs =
  List.exists
    (fun attr -> String.equal attr.Parsetree.attr_name.txt "loom.entry")
    attrs

let rec flatten_function_params (expr : Typedtree.expression) =
  match expr.exp_desc with
  | Texp_function (params, Tfunction_body body) ->
      let params', body' = flatten_function_params body in
      (params @ params', body')
  | _ -> ([], expr)

let classify_param primitive_ids (param : Typedtree.function_param) =
  match param.fp_kind with
  | Tparam_optional_default _ ->
      Diagnostic.raise_error ~loc:param.fp_loc "optional parameters are not supported in Loom entries"
  | Tparam_pat pat ->
      let name = pattern_name pat in
      let kind =
        match Loom_types.classify_type ~tensor_type:primitive_ids.tensor_type_path pat.pat_type with
        | Some Loom_types.LLFloat -> Loom_types.ScalarF32
        | Some Loom_types.LLTensor1F32 -> Loom_types.Tensor1F32
        | Some ty ->
            Diagnostic.raise_error ~loc:pat.pat_loc
              (Printf.sprintf "unsupported entry parameter type %s"
                 (Loom_types.stage_type_to_string ty))
        | None -> Diagnostic.raise_error ~loc:pat.pat_loc "unsupported entry parameter type"
      in
      { name; kind }

let entries_of_unit owner =
  owner.user_structure.str_items
  |> List.concat_map (fun item ->
         match item.str_desc with
         | Tstr_value (_, bindings) ->
             List.filter_map
               (fun binding ->
                 if not (has_loom_entry binding.vb_attributes) then None
                 else
                   let name = pattern_name binding.vb_pat in
                   let params, _ = flatten_function_params binding.vb_expr in
                   Some
                     { owner
                     ; name
                     ; loc = binding.vb_loc
                     ; binding
                     ; params = List.map (classify_param owner.primitive_ids) params })
               bindings
         | _ -> [])

let list_entries file = entries_of_unit (load_file file)

let find_entry file name =
  match List.find_opt (fun entry -> String.equal entry.name name) (list_entries file) with
  | Some entry -> entry
  | None ->
      Diagnostic.raise_error
        (Printf.sprintf "entry %S not found in %s" name file)
