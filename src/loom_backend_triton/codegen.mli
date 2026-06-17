type generated_module = {
  filename : string;
  source : string;
}

val generate :
  autotune:Autotune_config.t option ->
  program:Tensor_ir.program ->
  plan:Triton_plan.t ->
  generated_module
