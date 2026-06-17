type launch_bucket = Kernel_plan_common.launch_bucket = {
  max_n : int;
  block_size : int;
  num_warps : int;
}

type complexity_bucket = Kernel_plan_common.complexity_bucket =
  | Tiny
  | Small
  | Medium
  | Large

type producer_strategy = Kernel_plan_common.producer_strategy =
  | FusedProducer
  | ClonedProducer
  | MaterializedProducer

type reduction_strategy = Kernel_plan_common.reduction_strategy =
  | FixedStrategy
  | DirectReduction
  | SmallDirectReduction
  | SmallPartialReduction
  | SingleStagePartialReduction
  | MultiStageTreeReduction
  | TwoPhaseThresholdedReduction

type storage_class = Kernel_plan_common.storage_class =
  | OutputStorage
  | TemporaryStorage of int

type pointwise_class = Kernel_plan_common.pointwise_class =
  | PointwiseFastPath
  | GeneralPointwise

type reduction_source = Kernel_plan_common.reduction_source =
  | PlainInput of string
  | MappedInput of {
      inputs : (string * string) list;
      scalar_params : string list;
      body : Tensor_ir.scalar_expr;
    }

type cuda_pointwise_family =
  | GenericPointwiseKernel
  | SmallNPointwiseKernel
  | VectorizedPointwiseKernel

type cuda_reduction_execution =
  | WorkspaceTreeReduction
  | SingleBlockReduction
  | AtomicOutputReduction

type cuda_combine_family =
  | NoCombineKernel
  | SharedTreeCombine

type cuda_body_traits = string list

type elementwise_step = {
  node_id : int;
  kernel_name : string;
  output : string;
  inputs : (string * string) list;
  scalar_params : string list;
  block_size : int;
  num_warps : int;
  launch_buckets : launch_bucket list;
  plan_class : string;
  pointwise_class : pointwise_class;
  pointwise_family : cuda_pointwise_family;
  traits : cuda_body_traits;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type reduction_step = {
  node_id : int;
  kernel_name : string;
  combine_kernel_name : string;
  output : string;
  reduce_kind : Tensor_ir.reduce_kind;
  block_size : int;
  num_warps : int;
  launch_buckets : launch_bucket list;
  single_block_threshold : int option;
  small_reduction_threshold : int option;
  small_program_count : int option;
  source : reduction_source;
  reduction_family : string;
  reduction_class : string;
  strategy_kind : string;
  reduction_strategy : reduction_strategy;
  execution_family : cuda_reduction_execution;
  combine_family : cuda_combine_family;
  traits : cuda_body_traits;
  uses_workspace : bool;
  stage_count : int;
  stage_layout : string;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type step =
  | Elementwise of elementwise_step
  | Reduction of reduction_step

type t = {
  entry_name : string;
  steps : step list;
  result_name : string;
  temporary_count : int;
}

val of_program : ?optimizations:Optimizations.t -> Tensor_ir.program -> t
val to_yojson : t -> Yojson.Safe.t
val to_string : t -> string

val complexity_bucket_to_yojson : complexity_bucket -> Yojson.Safe.t
val producer_strategy_to_string : producer_strategy -> string
val pointwise_class_to_string : pointwise_class -> string
val reduction_strategy_to_string : reduction_strategy -> string
val pointwise_family_to_string : cuda_pointwise_family -> string
val reduction_execution_to_string : cuda_reduction_execution -> string
val combine_family_to_string : cuda_combine_family -> string
