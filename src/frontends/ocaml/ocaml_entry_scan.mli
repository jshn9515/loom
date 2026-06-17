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

val load_file : string -> compilation_unit
val list_entries : string -> entry list
val find_entry : string -> string -> entry
