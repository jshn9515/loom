include Optimization_types

let default_config_path = Optimization_config.default_config_path
let default_config = Optimization_config.default_config
let none = Optimization_config.none

let all_infos =
  [
    {
      id = ScalarConstFold;
      flag = "scalar-const-fold";
      name = "Scalar Constant Folding";
      stage = Normalization;
      purpose =
        "Fold scalar-only arithmetic, comparison, select, and nested let \
         expressions during FrontIR to LoomLambda normalization.";
      expected_results =
        "Smaller normalized programs, fewer scalar ops in tensor lambdas, and \
         cleaner downstream IR.";
      caveats =
        "Best run before other normalization passes; reduces some source-level \
         detail in debug dumps.";
    };
    {
      id = NormalizedDce;
      flag = "normalized-dce";
      name = "Normalized Dead Code Elimination";
      stage = Normalization;
      purpose =
        "Remove dead scalar lets and helper bindings after normalization and \
         local simplification.";
      expected_results =
        "Lower-noise LoomLambda output and fewer captured scalar parameters \
         downstream.";
      caveats =
        "Mostly an enabling pass; runtime wins are indirect unless redundant \
         helpers are common.";
    };
    {
      id = IfSimplify;
      flag = "if-simplify";
      name = "If Simplification";
      stage = Normalization;
      purpose =
        "Simplify scalar if-expressions with identical branches or \
         already-resolved conditions during normalization.";
      expected_results =
        "Cleaner normalized control flow and less scalar branching noise in \
         downstream IR.";
      caveats =
        "Works best after constant folding; benefits depend on source shape.";
    };
    {
      id = LetFloat;
      flag = "let-float";
      name = "Let Float";
      stage = Normalization;
      purpose =
        "Float common scalar lets out of branch-local structure when both \
         branches bind the same scalar value.";
      expected_results =
        "More shared scalar setup and less duplicated branch-local work.";
      caveats =
        "Only helps on specific source patterns with duplicated branch \
         bindings.";
    };
    {
      id = LambdaInlineSmall;
      flag = "lambda-inline-small";
      name = "Lambda Inline Small";
      stage = Normalization;
      purpose = "Inline small direct lambda applications before tensorization.";
      expected_results =
        "Fewer residual Apply nodes and better exposure of scalar/tensor \
         combinator structure.";
      caveats = "Guarded by a size threshold to avoid excessive code growth.";
    };
    {
      id = ScalarCse;
      flag = "scalar-cse";
      name = "Scalar Common Subexpression Elimination";
      stage = Normalization;
      purpose =
        "Share repeated scalar-only subexpressions in LoomLambda before \
         tensorization.";
      expected_results =
        "Smaller scalar bodies, more stable normalization, and better \
         exposure of repeated arithmetic.";
      caveats =
        "Most helpful on helper-expanded scalar code; can slightly increase \
         let structure in dumps.";
    };
    {
      id = ArithReassociate;
      flag = "arith-reassociate";
      name = "Arithmetic Reassociation";
      stage = Normalization;
      purpose =
        "Reassociate and canonicalize scalar arithmetic trees so equivalent \
         programs normalize to the same LoomLambda shape.";
      expected_results =
        "Better matching for scalar CSE and cleaner lowering into TensorIR.";
      caveats =
        "Restricted to pure scalar arithmetic; does not change tensor \
         combinator structure directly.";
    };
    {
      id = BranchHoist;
      flag = "branch-hoist";
      name = "Branch Hoist";
      stage = Normalization;
      purpose =
        "Hoist branch-invariant scalar work out of branch-local structure when \
         both arms compute the same scalar setup.";
      expected_results =
        "Less duplicated scalar work and better reuse across helper-expanded \
         programs.";
      caveats =
        "Conservative and best paired with scalar simplification passes.";
    };
    {
      id = TensorCanonicalize;
      flag = "tensor-canonicalize";
      name = "Tensor Canonicalization";
      stage = Tensorize;
      purpose =
        "Canonicalize scalar expressions and node metadata so equivalent \
         tensor programs lower to stable TensorIR.";
      expected_results =
        "More deterministic IR, improved golden stability, and better \
         interaction with later fusion and scheduling.";
      caveats =
        "Mainly a stabilization pass; direct runtime wins are usually small.";
    };
    {
      id = ReductionInputSimplify;
      flag = "reduction-input-simplify";
      name = "Reduction Input Simplification";
      stage = Tensorize;
      purpose =
        "Simplify scalar expressions feeding reduction-shaped tensor pipelines \
         before backend planning.";
      expected_results =
        "Smaller reduction bodies and less redundant scalar work around mapped \
         reductions.";
      caveats =
        "Most useful on reduction-heavy programs; usually neutral on pure \
         pointwise kernels.";
    };
    {
      id = ShapeUseSimplify;
      flag = "shape-use-simplify";
      name = "Shape Use Simplification";
      stage = Tensorize;
      purpose =
        "Drop unreachable TensorIR nodes after local rewrites and fusion.";
      expected_results =
        "Lower-noise TensorIR and fewer accidental temporaries reaching \
         backend planning.";
      caveats = "Primarily a cleanup pass; runtime wins are indirect.";
    };
    {
      id = MapChainCollapse;
      flag = "map-chain-collapse";
      name = "Map Chain Collapse";
      stage = Tensorize;
      purpose =
        "Collapse nested elementwise TensorIR producers into a single \
         elementwise body when shape and purity allow it.";
      expected_results =
        "Fewer temporary tensors and stronger pointwise fusion before backend \
         planning.";
      caveats =
        "Can increase scalar body complexity if used too aggressively.";
    };
    {
      id = ReduceMapFusion;
      flag = "reduce-map-fusion";
      name = "Reduce Map Fusion";
      stage = Tensorize;
      purpose =
        "Represent map-then-reduce pipelines as fused mapped reductions in \
         TensorIR instead of deferring the decision to backend planning.";
      expected_results =
        "Lower temporary traffic and better reduction planning on dot, norm, \
         and loss-style kernels.";
      caveats =
        "Makes reduction nodes richer and relies on good reduction planning \
         downstream.";
    };
    {
      id = SmallProducerClone;
      flag = "small-producer-clone";
      name = "Small Producer Clone";
      stage = Tensorize;
      purpose =
        "Clone very small pure elementwise producers into consumers when that \
         removes an intermediate tensor.";
      expected_results =
        "Fewer memory round-trips for tiny helper-produced intermediates.";
      caveats =
        "Should stay thresholded to avoid code growth and over-fusion.";
    };
    {
      id = ReductionBodyNormalize;
      flag = "reduction-body-normalize";
      name = "Reduction Body Normalize";
      stage = Tensorize;
      purpose =
        "Canonicalize mapped reduction bodies and metadata so equivalent \
         reduction pipelines share the same TensorIR form.";
      expected_results =
        "Better reduction matching and more stable planning for helper-shaped \
         reductions.";
      caveats =
        "Mostly an enabling pass; direct wins are indirect.";
    };
    {
      id = TensorCse;
      flag = "tensor-cse";
      name = "Tensor Common Subexpression Elimination";
      stage = Tensorize;
      purpose =
        "Merge structurally identical TensorIR nodes when they compute the \
         same value.";
      expected_results =
        "Less duplicate work and fewer redundant intermediates reaching the \
         planner.";
      caveats =
        "Requires careful structural matching and is primarily useful on \
         helper-expanded programs.";
    };
    {
      id = ElementwiseFusion;
      flag = "elementwise-fusion";
      name = "Elementwise Fusion";
      stage = Tensorize;
      purpose =
        "Fuse adjacent elementwise TensorIR nodes when producer and consumer \
         are shape-compatible and single-use.";
      expected_results =
        "Fewer temporaries, fewer launches, and lower memory traffic for \
         pointwise-heavy kernels.";
      caveats =
        "Can increase kernel body complexity and register pressure; some tasks \
         may regress.";
    };
    {
      id = ScalarHoist;
      flag = "scalar-hoist";
      name = "Scalar Hoist";
      stage = Tensorize;
      purpose =
        "Hoist tensor-lambda-invariant scalar lets out of tensor combinators \
         before tensorization.";
      expected_results =
        "Simpler elementwise bodies, fewer duplicated scalar computations, and \
         cleaner scalar capture sets.";
      caveats =
        "Most helpful when frontend code contains local scalar helpers or \
         repeated invariant lets.";
    };
    {
      id = MaterializationChoice;
      flag = "materialization-choice";
      name = "Materialization Choice";
      stage = Tensorize;
      purpose =
        "Choose whether small tensor producers should be fused, cloned, or \
         explicitly materialized from reuse and body-cost metadata.";
      expected_results =
        "Fewer harmful over-fusion cases on large pointwise pipelines and \
         better mapped-reduction producer choices.";
      caveats =
        "Heuristic and workload-sensitive; an overly conservative policy can \
         miss good fusion wins.";
    };
    {
      id = ReductionAccumulatorShape;
      flag = "reduction-accumulator-shape";
      name = "Reduction Accumulator Shape";
      stage = Tensorize;
      purpose =
        "Normalize mapped-reduction accumulator/update structure so \
         helper-expanded reductions converge before planning.";
      expected_results =
        "More stable mapped-reduction TensorIR and better matching across \
         expressive source forms.";
      caveats =
        "Primarily an enabling pass; most wins come from downstream planning.";
    };
    {
      id = ReductionReuseHoist;
      flag = "reduction-reuse-hoist";
      name = "Reduction Reuse Hoist";
      stage = Tensorize;
      purpose =
        "Hoist and simplify repeated scalar work inside mapped-reduction \
         bodies before backend planning.";
      expected_results =
        "Smaller reduction bodies and better reuse for weighted and branchy \
         mapped reductions.";
      caveats =
        "Best paired with reduction canonicalization and map-reduce fusion.";
    };
    {
      id = ReductionBodyCse;
      flag = "reduction-body-cse";
      name = "Reduction Body CSE";
      stage = Tensorize;
      purpose =
        "Collapse repeated mapped-reduction subexpressions into simpler scalar \
         forms before backend planning.";
      expected_results =
        "Lower scalar body cost for reduction kernels with repeated term \
         structure and more stable mapped-reduction normalization.";
      caveats =
        "Works through local scalar rewrites rather than full let-bound CSE, \
         so it helps only when repeated structure is simple enough to fold.";
    };
    {
      id = ReductionMaterializationChoice;
      flag = "reduction-materialization-choice";
      name = "Reduction Materialization Choice";
      stage = Tensorize;
      purpose =
        "Choose more carefully whether producers feeding reductions should be \
         fused, cloned, or materialized from reduction-shape and body-cost \
         metadata.";
      expected_results =
        "Fewer over-fused reduction inputs and better fixed-path behavior on \
         weighted, ratio, and helper-expanded reductions.";
      caveats =
        "Heuristic and workload-sensitive; too much conservatism can leave \
         good fusion wins on the table.";
    };
    {
      id = ReductionLateFusionGuard;
      flag = "reduction-late-fusion-guard";
      name = "Reduction Late Fusion Guard";
      stage = Tensorize;
      purpose =
        "Block reduction-directed fusion when the combined mapped body is too \
         costly or branch-heavy for the current fixed planning heuristics.";
      expected_results =
        "Fewer large reduction regressions caused by overly aggressive map-\
         into-reduce fusion.";
      caveats =
        "Can suppress some profitable fusion when the complexity threshold is \
         too low.";
    };
    {
      id = BranchAwareFusionGuard;
      flag = "branch-aware-fusion-guard";
      name = "Branch-Aware Fusion Guard";
      stage = Tensorize;
      purpose =
        "Block aggressive fusion when branch-heavy producer bodies are likely \
         to inflate large pointwise kernels.";
      expected_results =
        "Fewer branch-amplified regressions on workloads such as \
         affine-clamp-like and HFT signal transforms.";
      caveats =
        "Can disable useful fusion if the branchiness threshold is too low.";
    };
    {
      id = PointwiseMaterializationGuard;
      flag = "pointwise-materialization-guard";
      name = "Pointwise Materialization Guard";
      stage = Tensorize;
      purpose =
        "Prefer simpler materialized or lightly fused plans for expensive \
         division-heavy or branch-heavy pointwise producers.";
      expected_results =
        "Better large-size behavior on previously regressing pointwise \
         kernels.";
      caveats =
        "Mostly useful on large shapes; neutral or slightly negative on tiny \
         elementwise kernels.";
    };
    {
      id = ReductionPlanSpecialize;
      flag = "reduction-plan-specialize";
      name = "Reduction Plan Specialization";
      stage = Backend;
      purpose =
        "Pick stronger fixed launch heuristics for reduction kernels during \
         TensorIR to KernelPlan lowering.";
      expected_results =
        "Large fixed-path improvements on reduction-heavy workloads such as \
         dot, norms, and loss reductions.";
      caveats =
        "Overlaps with Triton autotune benefits and may not help very small \
         reductions.";
    };
    {
      id = ElementwisePlanSpecialize;
      flag = "elementwise-plan-specialize";
      name = "Elementwise Plan Specialization";
      stage = Backend;
      purpose =
        "Pick stronger fixed launch heuristics for elementwise kernels based \
         on input count and body complexity.";
      expected_results =
        "Incremental non-autotuned wins for simple pointwise kernels and \
         better starting points for autotune.";
      caveats =
        "Usually a smaller gain than fusion; can be neutral on \
         already-well-matched kernels.";
    };
    {
      id = ReductionPrecombine;
      flag = "reduction-precombine";
      name = "Reduction Precombine";
      stage = Backend;
      purpose =
        "Fuse a single-use elementwise producer into the first reduction stage \
         during KernelPlan lowering.";
      expected_results =
        "Eliminates an intermediate tensor and reduces launch/memory overhead \
         for map-then-reduce pipelines.";
      caveats =
        "Only applies when the mapped producer is single-use and body \
         complexity stays under a guard threshold.";
    };
    {
      id = ReductionTwoPhase;
      flag = "reduction-two-phase";
      name = "Reduction Two Phase";
      stage = Backend;
      purpose =
        "Switch small reductions to a direct single-block path while retaining \
         recursive multi-pass reduction for larger inputs.";
      expected_results =
        "Lower workspace pressure and better small/medium reduction latency.";
      caveats =
        "Threshold-sensitive; gains depend on workload size distribution.";
    };
    {
      id = LaunchBucketSpecialize;
      flag = "launch-bucket-specialize";
      name = "Launch Bucket Specialization";
      stage = Backend;
      purpose =
        "Choose fixed launch settings from size buckets instead of one global \
         default.";
      expected_results =
        "Better fixed-path performance across mixed small, medium, and large \
         workloads.";
      caveats =
        "Adds more fixed-policy surface and needs good bucket defaults to \
         avoid regressions.";
    };
    {
      id = ReductionClassify;
      flag = "reduction-classify";
      name = "Reduction Classification";
      stage = Backend;
      purpose =
        "Classify reductions by source form and scalar body complexity before \
         kernel planning.";
      expected_results =
        "Better generic planning decisions for plain, mapped, branchy, and \
         weighted reductions.";
      caveats =
        "Mainly an enabling pass; planning wins depend on downstream \
         heuristics.";
    };
    {
      id = ReductionTreePlan;
      flag = "reduction-tree-plan";
      name = "Reduction Tree Planning";
      stage = Backend;
      purpose =
        "Choose stronger generic reduction tree shapes from reduction class \
         and size bucket.";
      expected_results =
        "Lower steady-state latency on large reductions without backend-\
         specific tuning logic.";
      caveats =
        "Heuristic-driven and may need per-workload calibration.";
    };
    {
      id = ReductionSplitPlan;
      flag = "reduction-split-plan";
      name = "Reduction Split Planning";
      stage = Backend;
      purpose =
        "Choose direct, single-stage, or multi-stage reduction plans from \
         size and reduction class.";
      expected_results =
        "Better small/medium/large reduction behavior with clearer planning \
         metadata.";
      caveats =
        "Interacts with single-block thresholds and reduction tree choices.";
    };
    {
      id = ReductionStageSizing;
      flag = "reduction-stage-sizing";
      name = "Reduction Stage Sizing";
      stage = Backend;
      purpose =
        "Choose reduction stage counts from reduction class, body cost, and \
         configurable size thresholds rather than one coarse split.";
      expected_results =
        "More stable large-reduction performance and fewer mis-sized fixed \
         reduction plans.";
      caveats =
        "Threshold-heavy and may overlap with other reduction planning \
         passes.";
    };
    {
      id = ReductionStageBalance;
      flag = "reduction-stage-balance";
      name = "Reduction Stage Balance";
      stage = Backend;
      purpose =
        "Balance fixed reduction launch shape and stage layout from reduction \
         class, body cost, and repeated-subexpression pressure.";
      expected_results =
        "Better medium/large fixed-path reductions without relying on backend \
         autotune.";
      caveats =
        "Still heuristic-driven and may need retuning as the reduction suite \
         evolves.";
    };
    {
      id = ReductionPartialShapePack;
      flag = "reduction-partial-shape-pack";
      name = "Reduction Partial Shape Pack";
      stage = Backend;
      purpose =
        "Choose reduction launch shapes that pack intermediate partial buffers \
         more tightly for multi-stage reductions.";
      expected_results =
        "Lower partial-buffer footprint and better fixed-path reduction \
         latency on large inputs.";
      caveats =
        "Mostly helps staged reductions; plain small reductions may see no \
         change.";
    };
    {
      id = ReductionClassAwareBuckets;
      flag = "reduction-class-aware-buckets";
      name = "Reduction Class-Aware Buckets";
      stage = Backend;
      purpose =
        "Adjust fixed reduction launch buckets by mapped-reduction class \
         instead of reusing one bucket table for every reduction shape.";
      expected_results =
        "Better fixed launch choices across plain, weighted, branchy, and \
         ratio-style reductions.";
      caveats =
        "Adds more policy surface and needs conservative defaults to avoid \
         new regressions.";
    };
    {
      id = ReductionSmallPlanSpecialize;
      flag = "reduction-small-plan-specialize";
      name = "Reduction Small Plan Specialize";
      stage = Backend;
      purpose =
        "Use a dedicated small-reduction planning path for size-gated mapped, \
         weighted, and mapped-reuse reductions rather than routing them \
         through the generic partial-reduction plan.";
      expected_results =
        "Better small and medium reduction latency, especially around 131072 \
         elements, without disturbing large-input wins.";
      caveats =
        "Must stay narrowly size-gated; broad use can hurt large reductions \
         and unrelated workload classes.";
    };
    {
      id = ReductionBodyCanonicalize;
      flag = "reduction-body-canonicalize";
      name = "Reduction Body Canonicalize";
      stage = Tensorize;
      purpose =
        "Recognize and normalize common mapped-reduction scalar bodies so the \
         planner can classify weighted and repeated-difference reductions more \
         accurately.";
      expected_results =
        "Better reduction-family matching and improved planning for dot, \
         weighted-dot, and loss-style kernels.";
      caveats =
        "Should stay pattern-driven and reduction-specific rather than trying \
         to be a general algebra system.";
    };
    {
      id = BranchyReductionRescue;
      flag = "branchy-reduction-rescue";
      name = "Branchy Reduction Rescue";
      stage = Backend;
      purpose =
        "Use a branch-aware small-reduction plan for branchy mapped \
         reductions that remain slow under the generic planner.";
      expected_results =
        "Lower runtime on branch-heavy reductions such as \
         piecewise-weighted-dot at small and medium sizes.";
      caveats =
        "Should remain narrow so non-branchy reduction planning is not \
         perturbed.";
    };
    {
      id = TempLifetimePack;
      flag = "temp-lifetime-pack";
      name = "Temporary Lifetime Packing";
      stage = Backend;
      purpose =
        "Reuse temporary storage slots when plan step lifetimes do not \
         overlap.";
      expected_results =
        "Lower temporary count and less intermediate buffer pressure.";
      caveats =
        "Requires correct liveness modeling; mostly affects large pipelines.";
    };
    {
      id = StorageReusePack;
      flag = "storage-reuse-pack";
      name = "Storage Reuse Packing";
      stage = Backend;
      purpose =
        "Turn temporary lifetime analysis into real storage-name reuse inside \
         KernelPlan rather than metadata-only slot annotations.";
      expected_results =
        "Lower intermediate buffer count and more faithful temporary reuse in \
         generated backends.";
      caveats =
        "Requires careful input/output renaming so reused storage never \
         clobbers live values.";
    };
    {
      id = MultiUseFusionGuard;
      flag = "multi-use-fusion-guard";
      name = "Multi-Use Fusion Guard";
      stage = Backend;
      purpose =
        "Prevent planning decisions that over-fuse producers with reuse \
         patterns likely to regress.";
      expected_results =
        "More robust performance on workloads where aggressive fusion inflates \
         body complexity or duplicate work.";
      caveats =
        "Can disable some wins if the guard is too conservative.";
    };
    {
      id = PointwiseShapePlan;
      flag = "pointwise-shape-plan";
      name = "Pointwise Shape Planning";
      stage = Backend;
      purpose =
        "Choose better generic pointwise plan parameters from body complexity \
         and shape bucket.";
      expected_results =
        "Stronger fixed-path behavior for large pointwise kernels without \
         backend-specific scheduling.";
      caveats =
        "Incremental on already well-matched small kernels.";
    };
    {
      id = SharedBodyTraitAnalysis;
      flag = "shared-body-trait-analysis";
      name = "Shared Body Trait Analysis";
      stage = Backend;
      purpose =
        "Classify TensorIR scalar bodies into backend-neutral semantic traits \
         before TritonPlan and CudaPlan specialize them.";
      expected_results =
        "Keeps pattern recognition coherent across generated Triton and CUDA \
         while still allowing target-specific lowering choices.";
      caveats =
        "Diagnostic by itself; runtime changes require backend instruction \
         selection or launch planning flags.";
    };
    {
      id = TritonReductionInstructionSelect;
      flag = "triton-reduction-instruction-select";
      name = "Triton Reduction Instruction Selection";
      stage = Backend;
      purpose =
        "Use shared reduction traits to choose Triton-specific reduction \
         families and launch buckets for fixed-path generated kernels.";
      expected_results =
        "Lower small and medium reduction latency for generated Triton without \
         relying on autotune.";
      caveats =
        "Triton launch policy is size-sensitive and must be checked against \
         fixed external baselines.";
    };
    {
      id = TritonPointwiseInstructionSelect;
      flag = "triton-pointwise-instruction-select";
      name = "Triton Pointwise Instruction Selection";
      stage = Backend;
      purpose =
        "Use shared pointwise traits to choose Triton-specific elementwise \
         launch buckets for activation, clamp, and filter bodies.";
      expected_results =
        "Better generated Triton behavior on small pointwise kernels while \
         preserving generic pointwise defaults elsewhere.";
      caveats =
        "Should not duplicate autotune; fixed-path wins must be measured.";
    };
    {
      id = TritonSmallReductionTailTune;
      flag = "triton-small-reduction-tail-tune";
      name = "Triton Small Reduction Tail Tuning";
      stage = Backend;
      purpose =
        "Apply a narrower fixed-path launch policy for trait-classified small \
         Triton reductions that still lag handwritten fixed Triton kernels.";
      expected_results =
        "Lower tail latency for small dot, norm, weighted, and loss-style \
         reductions without changing large-size Triton behavior.";
      caveats =
        "Size-sensitive; must be validated against fixed external Triton and \
         CUDA baselines.";
    };
    {
      id = CudaReductionNormPlan;
      flag = "cuda-reduction-norm-plan";
      name = "CUDA Reduction Norm Planning";
      stage = Backend;
      purpose =
        "Use CUDA-specific reduction planning for norm-square reductions \
         instead of the generic fixed reduction planner.";
      expected_results =
        "Better stage and combine choices for l2-norm-style reductions and \
         affine norm-square reductions on the CUDA backend.";
      caveats =
        "Must remain mutually exclusive with overlapping generic reduction \
         split and staging heuristics.";
    };
    {
      id = CudaReductionDotPlan;
      flag = "cuda-reduction-dot-plan";
      name = "CUDA Reduction Dot Planning";
      stage = Backend;
      purpose =
        "Use CUDA-specific reduction planning for dot and dot-pipeline \
         reductions instead of the generic mapped reduction planner.";
      expected_results =
        "Better fixed launch shape and combine decisions for dot-style \
         reductions on CUDA.";
      caveats =
        "Should not overlap with the generic reduction split and stage \
         planning flags for the same profile.";
    };
    {
      id = CudaReductionWeightedProductPlan;
      flag = "cuda-reduction-weighted-product-plan";
      name = "CUDA Weighted Product Planning";
      stage = Backend;
      purpose =
        "Use a CUDA-specific planning path for weighted multiplicative \
         reductions such as weighted-dot and weighted-dot-pipeline.";
      expected_results =
        "Better family matching and fixed launch choices for weighted product \
         reductions on CUDA.";
      caveats =
        "Must stay distinct from delta-square and ratio reductions to avoid \
         broad regressions.";
    };
    {
      id = CudaReductionPipelinePlan;
      flag = "cuda-reduction-pipeline-plan";
      name = "CUDA Pipeline Reduction Planning";
      stage = Backend;
      purpose =
        "Recognize helper-expanded and pipeline reduction shapes and plan them \
         with CUDA-specific thresholds and launch families.";
      expected_results =
        "Lower latency on dot-pipeline, weighted-dot-pipeline, and similar \
         helper-expanded reductions.";
      caveats =
        "Should remain narrowly targeted so plain reductions keep their \
         current wins.";
    };
    {
      id = CudaPointwiseSmallShapePlan;
      flag = "cuda-pointwise-small-shape-plan";
      name = "CUDA Pointwise Small-Shape Planning";
      stage = Backend;
      purpose =
        "Use CUDA-specific small-N pointwise planning instead of the generic \
         pointwise shape planner for simple activation and filter kernels.";
      expected_results =
        "Better small and medium pointwise latency on CUDA without changing \
         Triton behavior.";
      caveats =
        "Should remain mutually exclusive with the generic pointwise shape and \
         launch planning path.";
    };
    {
      id = CudaBodyTraitAnalysis;
      flag = "cuda-body-trait-analysis";
      name = "CUDA Body Trait Analysis";
      stage = Backend;
      purpose =
        "Classify TensorIR scalar bodies into CUDA-specific target traits \
         before instruction selection.";
      expected_results =
        "More precise CUDA plans for mixed compute bodies than a single broad \
         reduction family can provide.";
      caveats =
        "Diagnostic by itself; runtime changes require target instruction \
         selection flags.";
    };
    {
      id = CudaReductionInstructionSelect;
      flag = "cuda-reduction-instruction-select";
      name = "CUDA Reduction Instruction Selection";
      stage = Backend;
      purpose =
        "Choose CUDA reduction execution policy from target traits, including \
         size-sensitive direct, workspace, and atomic paths.";
      expected_results =
        "Narrower CUDA reduction choices that avoid profile-wide family \
         regressions.";
      caveats =
        "Must be guarded by full-suite regression checks because launch policy \
         is highly size-sensitive.";
    };
    {
      id = CudaSpecializedCombine;
      flag = "cuda-specialized-combine";
      name = "CUDA Specialized Combine";
      stage = Backend;
      purpose =
        "Allow family-specific CUDA combine kernels instead of one generic \
         shared-tree combine template.";
      expected_results =
        "Lower late-stage overhead for reduction families with distinctive \
         partial-result behavior.";
      caveats =
        "Can increase generated code size and should be activated only for \
         measured wins.";
    };
    {
      id = CudaPointwiseInstructionSelect;
      flag = "cuda-pointwise-instruction-select";
      name = "CUDA Pointwise Instruction Selection";
      stage = Backend;
      purpose =
        "Choose CUDA pointwise launch and kernel family from target traits such \
         as activation, affine update, thresholding, and filter bodies.";
      expected_results =
        "Avoid all-or-nothing small-shape routing for mixed pointwise kernels.";
      caveats =
        "Needs careful guard coverage because small kernels are dominated by \
         launch and synchronization costs.";
    };
    {
      id = CudaPointwiseMediumPlan;
      flag = "cuda-pointwise-medium-plan";
      name = "CUDA Medium Pointwise Planning";
      stage = Backend;
      purpose =
        "Route selected medium-size generic branchy threshold/clip CUDA \
         pointwise bodies through scalar loop codegen with a handwritten-style \
         block cap.";
      expected_results =
        "Close remaining medium-size affine clamp and soft-threshold gaps \
         without disturbing simple activation vector wins.";
      caveats =
        "CUDA-only and size-sensitive; must be accepted only after \
         non-held-out regression gates pass.";
    };
    {
      id = CudaPointwisePredicatedSelect;
      flag = "cuda-pointwise-predicated-select";
      name = "CUDA Predicated Select Lowering";
      stage = Backend;
      purpose =
        "Prefer CUDA math/select idioms such as fminf/fmaxf for clamp and \
         threshold-shaped pointwise bodies instead of branchy per-element \
         updates.";
      expected_results =
        "Lower branch overhead on ReLU, clamp, soft-threshold, and related \
         pointwise kernels.";
      caveats =
        "Must remain pattern-based so mixed branch-heavy code does not lose \
         short-circuit behavior unexpectedly.";
    };
    {
      id = CudaPointwiseAffineVectorTail;
      flag = "cuda-pointwise-affine-vector-tail";
      name = "CUDA Affine Vector Tail Tuning";
      stage = Backend;
      purpose =
        "Use CUDA-specific vector/tail routing for affine vector updates while \
         leaving ratio and branch-heavy bodies on their existing paths.";
      expected_results =
        "Reduce small and medium AXPY-style launch/body overhead without \
         changing reduction behavior.";
      caveats =
        "Applies only to trait-classified affine vector updates.";
    };
    {
      id = CudaBookFilterRegisterPlan;
      flag = "cuda-book-filter-register-plan";
      name = "CUDA Book Filter Register Planning";
      stage = Backend;
      purpose =
        "Keep complex order-book/filter pointwise bodies on scalar codegen \
         when vector lane duplication is likely to increase register pressure.";
      expected_results =
        "Improve large branch-heavy HFT pointwise tails while preserving \
         simpler vectorized pointwise wins.";
      caveats =
        "High-risk and should be enabled only when focused benchmarks show a \
         clear win.";
    };
    {
      id = CudaPointwiseVectorize;
      flag = "cuda-pointwise-vectorize";
      name = "CUDA Pointwise Vectorization";
      stage = Backend;
      purpose =
        "Emit CUDA-native float4 pointwise kernels for alignment-safe tensor \
         maps instead of scalar elementwise loops.";
      expected_results =
        "Higher memory throughput and lower loop overhead on bandwidth-bound \
         pointwise kernels.";
      caveats =
        "Requires CUDA backend artifact inspection and verification because \
         not every scalar body benefits from vectorized lanes.";
    };
    {
      id = CudaPointwiseTailTune;
      flag = "cuda-pointwise-tail-tune";
      name = "CUDA Pointwise Tail Tuning";
      stage = Backend;
      purpose =
        "Use shared pointwise body traits to select CUDA-specific scalar, \
         small-N, and vectorized pointwise families for the remaining \
         bandwidth and branch tails.";
      expected_results =
        "Close small and medium pointwise gaps without changing the shared \
         KernelPlan architecture or Triton behavior.";
      caveats =
        "CUDA-only and intentionally trait-sensitive; accept only after \
         focused pass comparisons against the previous fixed CUDA profile.";
    };
    {
      id = CudaPointwiseSelectedTailPlan;
      flag = "cuda-pointwise-selected-tail-plan";
      name = "CUDA Selected Pointwise Tail Planning";
      stage = Backend;
      purpose =
        "Apply CUDA-specific, trait-selected expression and launch choices for \
         pointwise bodies whose best generated plans differ from the shared \
         generic policy.";
      expected_results =
        "Reduce remaining affine vector update pointwise gaps without changing \
         Triton or reduction behavior; selected non-branching ratio-book \
         pointwise bodies may also use CUDA fast divide lowering.";
      caveats =
        "CUDA-only and deliberately narrow; every selected trait must keep a \
         same-run former-self baseline and non-held-out regression guard.";
    };
    {
      id = CudaReductionShuffle;
      flag = "cuda-reduction-shuffle";
      name = "CUDA Shuffle Reduction";
      stage = Backend;
      purpose =
        "Use warp shuffle based block reductions in generated CUDA instead of \
         a pure shared-memory tree.";
      expected_results =
        "Lower synchronization and shared-memory pressure for mapped and plain \
         reductions.";
      caveats =
        "Must preserve max and sum semantics and be benchmarked against the \
         previous warp-tail shared-memory reduction.";
    };
    {
      id = CudaReductionTailTune;
      flag = "cuda-reduction-tail-tune";
      name = "CUDA Reduction Tail Tuning";
      stage = Backend;
      purpose =
        "Route selected branchy, robust, and ratio-style CUDA reductions \
         through the existing atomic-output path when the workspace/combine \
         path dominates tail latency.";
      expected_results =
        "Reduce the remaining medium-size CUDA reduction tail without changing \
         the CUDA backend architecture or public artifact contract.";
      caveats =
        "Atomic accumulation is size- and body-shape-sensitive; accept only \
         when focused and full-suite regression gates pass.";
    };
    {
      id = CudaReductionMediumTailPlan;
      flag = "cuda-reduction-medium-tail-plan";
      name = "CUDA Medium Reduction Tail Planning";
      stage = Backend;
      purpose =
        "Extend CUDA reduction tail routing for medium branchy reductions \
         whose workspace/combine overhead dominates fixed-path latency.";
      expected_results =
        "Reduce remaining medium-size robust and branchy reduction gaps \
         without changing the reduction architecture.";
      caveats =
        "Atomic routing is workload-sensitive; must be validated across the \
         non-held-out suite before enabling publicly.";
    };
    {
      id = CudaAsyncWrapperReturn;
      flag = "cuda-async-wrapper-return";
      name = "CUDA Async Wrapper Return";
      stage = Backend;
      purpose =
        "Avoid a host-side cudaDeviceSynchronize in generated wrappers when \
         the caller already owns synchronization for timing and correctness.";
      expected_results =
        "Fairer comparison with hand-written CUDA wrappers and much lower \
         launch-dominated latency.";
      caveats =
        "Generated wrappers return after enqueueing work; callers that need \
         host-visible completion must synchronize explicitly.";
    };
    {
      id = CudaSelectedProfile;
      flag = "cuda-selected-profile";
      name = "CUDA Selected Profile";
      stage = Backend;
      purpose =
        "Mark a measured subset profile used to showcase the best discovered \
         CUDA fixed-path optimization set.";
      expected_results =
        "Expose selected CUDA results separately from the all-promoted CUDA \
         profile.";
      caveats =
        "Should not imply a semantic change by itself.";
    };
  ]

let info id =
  match List.find_opt (fun item -> item.id = id) all_infos with
  | Some item -> item
  | None -> invalid_arg "missing optimization info"

let flag id = (info id).flag

let parse_flag name =
  all_infos
  |> List.find_opt (fun item -> String.equal item.flag name)
  |> Option.map (fun item -> item.id)

let full = Optimization_config.full all_infos
let enable = Optimization_config.enable
let disable = Optimization_config.disable
let enabled = Optimization_config.enabled

let expect_assoc = function
  | `Assoc fields -> fields
  | _ ->
      Diagnostic.raise_error "expected JSON object in Loom optimization config"

let expect_int field = function
  | `Int value -> value
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected integer field %s in Loom optimization config"
           field)

let expect_list field = function
  | `List items -> items
  | _ ->
      Diagnostic.raise_error
        (Printf.sprintf "expected list field %s in Loom optimization config"
           field)

let field fields name =
  match List.assoc_opt name fields with
  | Some value -> value
  | None ->
      Diagnostic.raise_error
        (Printf.sprintf "missing Loom optimization config field %s" name)

let field_opt fields name = List.assoc_opt name fields

let launch_bucket_of_yojson json =
  let fields = expect_assoc json in
  {
    max_n = field fields "max_n" |> expect_int "max_n";
    block_size = field fields "block_size" |> expect_int "block_size";
    num_warps = field fields "num_warps" |> expect_int "num_warps";
  }

let config_of_yojson source_path json =
  let fields = expect_assoc json in
  {
    version = field fields "version" |> expect_int "version";
    small_inline_nodes =
      field fields "small_inline_nodes" |> expect_int "small_inline_nodes";
    small_clone_nodes =
      (match field_opt fields "small_clone_nodes" with
      | Some value -> expect_int "small_clone_nodes" value
      | None -> default_config.small_clone_nodes);
    reduction_precombine_max_body_complexity =
      field fields "reduction_precombine_max_body_complexity"
      |> expect_int "reduction_precombine_max_body_complexity";
    single_block_reduction_threshold =
      field fields "single_block_reduction_threshold"
      |> expect_int "single_block_reduction_threshold";
    reduction_stage_split_threshold =
      (match field_opt fields "reduction_stage_split_threshold" with
      | Some value -> expect_int "reduction_stage_split_threshold" value
      | None -> default_config.reduction_stage_split_threshold);
    branchy_fusion_complexity_threshold =
      (match field_opt fields "branchy_fusion_complexity_threshold" with
      | Some value -> expect_int "branchy_fusion_complexity_threshold" value
      | None -> default_config.branchy_fusion_complexity_threshold);
    pointwise_materialization_complexity_threshold =
      (match field_opt fields "pointwise_materialization_complexity_threshold" with
      | Some value -> expect_int "pointwise_materialization_complexity_threshold" value
      | None -> default_config.pointwise_materialization_complexity_threshold);
    materialize_multi_use_complexity_threshold =
      (match field_opt fields "materialize_multi_use_complexity_threshold" with
      | Some value -> expect_int "materialize_multi_use_complexity_threshold" value
      | None -> default_config.materialize_multi_use_complexity_threshold);
    reduction_stage_medium_threshold =
      (match field_opt fields "reduction_stage_medium_threshold" with
      | Some value -> expect_int "reduction_stage_medium_threshold" value
      | None -> default_config.reduction_stage_medium_threshold);
    reduction_stage_large_threshold =
      (match field_opt fields "reduction_stage_large_threshold" with
      | Some value -> expect_int "reduction_stage_large_threshold" value
      | None -> default_config.reduction_stage_large_threshold);
    small_direct_reduction_threshold =
      (match field_opt fields "small_direct_reduction_threshold" with
      | Some value -> expect_int "small_direct_reduction_threshold" value
      | None -> default_config.small_direct_reduction_threshold);
    small_partial_reduction_threshold =
      (match field_opt fields "small_partial_reduction_threshold" with
      | Some value -> expect_int "small_partial_reduction_threshold" value
      | None -> default_config.small_partial_reduction_threshold);
    small_reduction_program_count =
      (match field_opt fields "small_reduction_program_count" with
      | Some value -> expect_int "small_reduction_program_count" value
      | None -> default_config.small_reduction_program_count);
    branchy_small_reduction_threshold =
      (match field_opt fields "branchy_small_reduction_threshold" with
      | Some value -> expect_int "branchy_small_reduction_threshold" value
      | None -> default_config.branchy_small_reduction_threshold);
    branchy_small_reduction_program_count =
      (match field_opt fields "branchy_small_reduction_program_count" with
      | Some value -> expect_int "branchy_small_reduction_program_count" value
      | None -> default_config.branchy_small_reduction_program_count);
    reduction_body_cse_min_occurrences =
      (match field_opt fields "reduction_body_cse_min_occurrences" with
      | Some value -> expect_int "reduction_body_cse_min_occurrences" value
      | None -> default_config.reduction_body_cse_min_occurrences);
    reduction_materialize_complexity_threshold =
      (match field_opt fields "reduction_materialize_complexity_threshold" with
      | Some value -> expect_int "reduction_materialize_complexity_threshold" value
      | None -> default_config.reduction_materialize_complexity_threshold);
    reduction_late_fusion_complexity_threshold =
      (match field_opt fields "reduction_late_fusion_complexity_threshold" with
      | Some value -> expect_int "reduction_late_fusion_complexity_threshold" value
      | None -> default_config.reduction_late_fusion_complexity_threshold);
    elementwise_buckets =
      field fields "elementwise_buckets"
      |> expect_list "elementwise_buckets"
      |> List.map launch_bucket_of_yojson;
    reduction_buckets =
      field fields "reduction_buckets"
      |> expect_list "reduction_buckets"
      |> List.map launch_bucket_of_yojson;
    source_path;
  }

let load_config = Optimization_config.load_config

let of_cli_flags ~config_path ~enable_flags ~disable_flags =
  Optimization_config.of_cli_flags ~parse_flag ~config_path ~enable_flags
    ~disable_flags

let stage_to_string = function
  | Normalization -> "normalization"
  | Tensorize -> "tensorize"
  | Backend -> "backend"

let to_string_list config =
  Optimization_config.to_string_list ~all_infos config

let launch_bucket_to_yojson bucket =
  `Assoc
    [
      ("max_n", `Int bucket.max_n);
      ("block_size", `Int bucket.block_size);
      ("num_warps", `Int bucket.num_warps);
    ]

let config_to_yojson config =
  `Assoc
    [
      ("version", `Int config.version);
      ("small_inline_nodes", `Int config.small_inline_nodes);
      ("small_clone_nodes", `Int config.small_clone_nodes);
      ( "reduction_precombine_max_body_complexity",
        `Int config.reduction_precombine_max_body_complexity );
      ( "single_block_reduction_threshold",
        `Int config.single_block_reduction_threshold );
      ( "reduction_stage_split_threshold",
        `Int config.reduction_stage_split_threshold );
      ( "branchy_fusion_complexity_threshold",
        `Int config.branchy_fusion_complexity_threshold );
      ( "pointwise_materialization_complexity_threshold",
        `Int config.pointwise_materialization_complexity_threshold );
      ( "materialize_multi_use_complexity_threshold",
        `Int config.materialize_multi_use_complexity_threshold );
      ( "reduction_stage_medium_threshold",
        `Int config.reduction_stage_medium_threshold );
      ( "reduction_stage_large_threshold",
        `Int config.reduction_stage_large_threshold );
      ( "small_direct_reduction_threshold",
        `Int config.small_direct_reduction_threshold );
      ( "small_partial_reduction_threshold",
        `Int config.small_partial_reduction_threshold );
      ( "small_reduction_program_count",
        `Int config.small_reduction_program_count );
      ( "branchy_small_reduction_threshold",
        `Int config.branchy_small_reduction_threshold );
      ( "branchy_small_reduction_program_count",
        `Int config.branchy_small_reduction_program_count );
      ( "reduction_body_cse_min_occurrences",
        `Int config.reduction_body_cse_min_occurrences );
      ( "reduction_materialize_complexity_threshold",
        `Int config.reduction_materialize_complexity_threshold );
      ( "reduction_late_fusion_complexity_threshold",
        `Int config.reduction_late_fusion_complexity_threshold );
      ( "elementwise_buckets",
        `List (List.map launch_bucket_to_yojson config.elementwise_buckets) );
      ( "reduction_buckets",
        `List (List.map launch_bucket_to_yojson config.reduction_buckets) );
      ( "source_path",
        match config.source_path with
        | Some path -> `String path
        | None -> `Null );
    ]

let to_yojson config = Optimization_config.to_yojson ~all_infos config

let render_cli_list () =
  let render_info item =
    String.concat "\n"
      [
        Printf.sprintf "Optimization: [%s]" item.name;
        Printf.sprintf "Flag to enable: [--enable-opt %s]" item.flag;
        "Purpose:";
        item.purpose;
        "Expected results:";
        item.expected_results;
        "Caveats:";
        item.caveats;
        Printf.sprintf "Stage: [%s]" (stage_to_string item.stage);
      ]
  in
  String.concat "\n\n" (List.map render_info all_infos) ^ "\n"
