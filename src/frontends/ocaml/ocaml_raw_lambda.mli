type lowered_entry = {
  entry : Ocaml_entry_scan.entry;
  raw_lambda : Lambda.lambda;
}

val lower_entry : Ocaml_entry_scan.entry -> lowered_entry
val raw_lambda_to_string : lowered_entry -> string
