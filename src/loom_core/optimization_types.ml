type stage = Normalization | Tensorize | Backend

type id =
  | ScalarConstFold
  | NormalizedDce
  | IfSimplify
  | LetFloat
  | LambdaInlineSmall
  | ScalarCse
  | ArithReassociate
  | BranchHoist
  | TensorCanonicalize
  | ReductionInputSimplify
  | ShapeUseSimplify
  | MapChainCollapse
  | ReduceMapFusion
  | SmallProducerClone
  | ReductionBodyNormalize
  | TensorCse
  | ElementwiseFusion
  | ScalarHoist
  | MaterializationChoice
  | ReductionAccumulatorShape
  | ReductionReuseHoist
  | ReductionBodyCse
  | ReductionMaterializationChoice
  | ReductionLateFusionGuard
  | BranchAwareFusionGuard
  | PointwiseMaterializationGuard
  | ReductionPlanSpecialize
  | ElementwisePlanSpecialize
  | ReductionPrecombine
  | ReductionTwoPhase
  | LaunchBucketSpecialize
  | ReductionClassify
  | ReductionTreePlan
  | ReductionSplitPlan
  | ReductionStageSizing
  | ReductionStageBalance
  | ReductionPartialShapePack
  | ReductionClassAwareBuckets
  | ReductionSmallPlanSpecialize
  | ReductionBodyCanonicalize
  | BranchyReductionRescue
  | TempLifetimePack
  | StorageReusePack
  | MultiUseFusionGuard
  | PointwiseShapePlan
  | SharedBodyTraitAnalysis
  | TritonReductionInstructionSelect
  | TritonPointwiseInstructionSelect
  | TritonSmallReductionTailTune
  | CudaReductionNormPlan
  | CudaReductionDotPlan
  | CudaReductionWeightedProductPlan
  | CudaReductionPipelinePlan
  | CudaPointwiseSmallShapePlan
  | CudaBodyTraitAnalysis
  | CudaReductionInstructionSelect
  | CudaSpecializedCombine
  | CudaPointwiseInstructionSelect
  | CudaPointwiseMediumPlan
  | CudaPointwisePredicatedSelect
  | CudaPointwiseAffineVectorTail
  | CudaBookFilterRegisterPlan
  | CudaPointwiseVectorize
  | CudaPointwiseTailTune
  | CudaPointwiseSelectedTailPlan
  | CudaReductionShuffle
  | CudaReductionTailTune
  | CudaReductionMediumTailPlan
  | CudaAsyncWrapperReturn
  | CudaSelectedProfile

type info = {
  id : id;
  flag : string;
  name : string;
  stage : stage;
  purpose : string;
  expected_results : string;
  caveats : string;
}

type launch_bucket = { max_n : int; block_size : int; num_warps : int }

type config = {
  version : int;
  small_inline_nodes : int;
  small_clone_nodes : int;
  reduction_precombine_max_body_complexity : int;
  single_block_reduction_threshold : int;
  reduction_stage_split_threshold : int;
  branchy_fusion_complexity_threshold : int;
  pointwise_materialization_complexity_threshold : int;
  materialize_multi_use_complexity_threshold : int;
  reduction_stage_medium_threshold : int;
  reduction_stage_large_threshold : int;
  small_direct_reduction_threshold : int;
  small_partial_reduction_threshold : int;
  small_reduction_program_count : int;
  branchy_small_reduction_threshold : int;
  branchy_small_reduction_program_count : int;
  reduction_body_cse_min_occurrences : int;
  reduction_materialize_complexity_threshold : int;
  reduction_late_fusion_complexity_threshold : int;
  elementwise_buckets : launch_bucket list;
  reduction_buckets : launch_bucket list;
  source_path : string option;
}

module Id_ord = struct
  type t = id

  let compare = compare
end

module Id_set = Set.Make (Id_ord)

type t = { enabled : Id_set.t; config : config }
