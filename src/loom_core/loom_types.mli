type stage_type =
  | LLFloat
  | LLBool
  | LLTensor1F32
  | LLUnit
  | LLTuple of stage_type list
  | LLFun of stage_type list * stage_type

type entry_param_kind =
  | ScalarF32
  | Tensor1F32

val stage_type_to_string : stage_type -> string
val stage_type_of_string : string -> stage_type option
val entry_param_kind_to_string : entry_param_kind -> string
val classify_type : tensor_type:Path.t -> Types.type_expr -> stage_type option
