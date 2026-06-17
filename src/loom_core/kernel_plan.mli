type launch_bucket = { max_n : int; block_size : int; num_warps : int }

type complexity_bucket =
  | Tiny
  | Small
  | Medium
  | Large

type producer_strategy =
  | FusedProducer
  | ClonedProducer
  | MaterializedProducer

type reduction_strategy =
  | FixedStrategy
  | DirectReduction
  | SmallDirectReduction
  | SmallPartialReduction
  | SingleStagePartialReduction
  | MultiStageTreeReduction
  | TwoPhaseThresholdedReduction

type storage_class =
  | OutputStorage
  | TemporaryStorage of int

type pointwise_class =
  | PointwiseFastPath
  | GeneralPointwise

type body_traits = string list

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
  traits : body_traits;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type reduction_source =
  | PlainInput of string
  | MappedInput of {
      inputs : (string * string) list;
      scalar_params : string list;
      body : Tensor_ir.scalar_expr;
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
  traits : body_traits;
  strategy_kind : string;
  reduction_strategy : reduction_strategy;
  stage_count : int;
  stage_layout : string;
  complexity_bucket : complexity_bucket;
  producer_strategy : producer_strategy;
  storage_class : storage_class;
  temp_slot : int option;
}

type step = Elementwise of elementwise_step | Reduction of reduction_step

type t = {
  entry_name : string;
  steps : step list;
  result_name : string;
  temporary_count : int;
}

val of_program : ?optimizations:Optimizations.t -> Tensor_ir.program -> t
val to_yojson : t -> Yojson.Safe.t
val to_string : t -> string
val source_name_of_ref : t -> Tensor_ir.value_ref -> string
