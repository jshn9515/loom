let ensure_defined defined = function
  | Tensor_ir.ParamRef _ -> ()
  | Tensor_ir.NodeRef id ->
      if not (List.mem id defined) then
        Diagnostic.raise_error (Printf.sprintf "internal error: reference to undefined node %d" id)

let program (program : Tensor_ir.program) =
  let seen = ref [] in
  List.iter
    (function
      | Tensor_ir.Elementwise1D { id; inputs; _ } ->
          List.iter
            (fun (binding : Tensor_ir.input_binding) ->
              ensure_defined !seen binding.source)
            inputs;
          seen := id :: !seen
      | Tensor_ir.Reduce1D { id; source; _ } ->
          begin
            match source with
            | Tensor_ir.PlainInput input -> ensure_defined !seen input
            | MappedInput { inputs; scalar_params; body } ->
                List.iter
                  (fun (binding : Tensor_ir.input_binding) ->
                    ensure_defined !seen binding.source)
                  inputs;
                let allowed =
                  inputs
                  |> List.map (fun (binding : Tensor_ir.input_binding) -> binding.name)
                  |> List.sort_uniq String.compare
                in
                let referenced =
                  Tensor_ir.scalar_expr_free_vars body |> List.sort_uniq String.compare
                in
                List.iter
                  (fun name ->
                    if
                      not
                        (List.mem name allowed || List.mem name scalar_params)
                    then
                      Diagnostic.raise_error
                        (Printf.sprintf
                           "internal error: reduction body references unknown \
                            scalar %s"
                           name))
                  referenced
          end;
          seen := id :: !seen)
    program.nodes;
  match program.result with
  | Tensor_ir.TensorResult value | Tensor_ir.ScalarResult value -> ensure_defined !seen value
