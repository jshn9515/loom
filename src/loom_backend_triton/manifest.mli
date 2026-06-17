val render_front_ir_manifest :
  front_entry:Front_ir.entry ->
  lambda_entry:LoomLambda.entry ->
  program:Tensor_ir.program ->
  kernel_plan:Kernel_plan.t ->
  backend_plan:Triton_plan.t ->
  optimizations:Optimizations.t ->
  string

val render_manifest :
  program:Tensor_ir.program ->
  plan:Triton_plan.t ->
  autotune:Autotune_config.t option ->
  generated_files:string list ->
  optimizations:Optimizations.t ->
  string

val render_report :
  front_entry:Front_ir.entry ->
  lambda_entry:LoomLambda.entry ->
  program:Tensor_ir.program ->
  plan:Triton_plan.t ->
  autotune:Autotune_config.t option ->
  optimizations:Optimizations.t ->
  string
