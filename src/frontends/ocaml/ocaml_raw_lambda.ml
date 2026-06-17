type lowered_entry = {
  entry : Ocaml_entry_scan.entry;
  raw_lambda : Lambda.lambda;
}

let lower_entry entry =
  let scopes = Debuginfo.Scoped_location.empty_scopes in
  let raw_lambda = Translcore.transl_exp ~scopes entry.Ocaml_entry_scan.binding.vb_expr in
  let raw_lambda = Simplif.simplify_lambda raw_lambda in
  { entry; raw_lambda }

let raw_lambda_to_string lowered =
  Format.asprintf "%a@." Printlambda.lambda lowered.raw_lambda
