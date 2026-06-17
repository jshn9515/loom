type param_summary = {
  name : string;
  kind : Loom_types.entry_param_kind;
}

type entry_summary = {
  name : string;
  params : param_summary list;
}

val list_entries : string -> entry_summary list
val import_entry : string -> string -> Front_ir.entry
