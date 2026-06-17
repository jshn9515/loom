# Loom Optimizations

Loom exposes optimization passes as explicit flags so they can be toggled
individually during `compile`, `package`, and experiment runs.

Optimization: [Scalar Constant Folding]
Flag to enable: [--enable-opt scalar-const-fold]
Purpose:
Fold scalar-only arithmetic, comparison, select, and nested let expressions during FrontIR to LoomLambda normalization.
Expected results:
Smaller normalized programs, fewer scalar ops in tensor lambdas, and cleaner downstream IR.
Caveats:
Best run before other normalization passes; reduces some source-level detail in debug dumps.
Stage: [normalization]

Optimization: [Normalized Dead Code Elimination]
Flag to enable: [--enable-opt normalized-dce]
Purpose:
Remove dead scalar lets and helper bindings after normalization and local simplification.
Expected results:
Lower-noise LoomLambda output and fewer captured scalar parameters downstream.
Caveats:
Mostly an enabling pass; runtime wins are indirect unless redundant helpers are common.
Stage: [normalization]

Optimization: [If Simplification]
Flag to enable: [--enable-opt if-simplify]
Purpose:
Simplify scalar if-expressions with identical branches or already-resolved conditions during normalization.
Expected results:
Cleaner normalized control flow and less scalar branching noise in downstream IR.
Caveats:
Works best after constant folding; benefits depend on source shape.
Stage: [normalization]

Optimization: [Let Float]
Flag to enable: [--enable-opt let-float]
Purpose:
Float common scalar lets out of branch-local structure when both branches bind the same scalar value.
Expected results:
More shared scalar setup and less duplicated branch-local work.
Caveats:
Only helps on specific source patterns with duplicated branch bindings.
Stage: [normalization]

Optimization: [Lambda Inline Small]
Flag to enable: [--enable-opt lambda-inline-small]
Purpose:
Inline small direct lambda applications before tensorization.
Expected results:
Fewer residual Apply nodes and better exposure of scalar/tensor combinator structure.
Caveats:
Guarded by a size threshold to avoid excessive code growth.
Stage: [normalization]

Optimization: [Scalar Common Subexpression Elimination]
Flag to enable: [--enable-opt scalar-cse]
Purpose:
Share repeated scalar-only subexpressions in LoomLambda before tensorization.
Expected results:
Smaller scalar bodies, more stable normalization, and better exposure of repeated arithmetic.
Caveats:
Most helpful on helper-expanded scalar code; can slightly increase let structure in dumps.
Stage: [normalization]

Optimization: [Arithmetic Reassociation]
Flag to enable: [--enable-opt arith-reassociate]
Purpose:
Reassociate and canonicalize scalar arithmetic trees so equivalent programs normalize to the same LoomLambda shape.
Expected results:
Better matching for scalar CSE and cleaner lowering into TensorIR.
Caveats:
Restricted to pure scalar arithmetic; does not change tensor combinator structure directly.
Stage: [normalization]

Optimization: [Branch Hoist]
Flag to enable: [--enable-opt branch-hoist]
Purpose:
Hoist branch-invariant scalar work out of branch-local structure when both arms compute the same scalar setup.
Expected results:
Less duplicated scalar work and better reuse across helper-expanded programs.
Caveats:
Conservative and best paired with scalar simplification passes.
Stage: [normalization]

Optimization: [Tensor Canonicalization]
Flag to enable: [--enable-opt tensor-canonicalize]
Purpose:
Canonicalize scalar expressions and node metadata so equivalent tensor programs lower to stable TensorIR.
Expected results:
More deterministic IR, improved golden stability, and better interaction with later fusion and scheduling.
Caveats:
Mainly a stabilization pass; direct runtime wins are usually small.
Stage: [tensorize]

Optimization: [Reduction Input Simplification]
Flag to enable: [--enable-opt reduction-input-simplify]
Purpose:
Simplify scalar expressions feeding reduction-shaped tensor pipelines before backend planning.
Expected results:
Smaller reduction bodies and less redundant scalar work around mapped reductions.
Caveats:
Most useful on reduction-heavy programs; usually neutral on pure pointwise kernels.
Stage: [tensorize]

Optimization: [Shape Use Simplification]
Flag to enable: [--enable-opt shape-use-simplify]
Purpose:
Drop unreachable TensorIR nodes after local rewrites and fusion.
Expected results:
Lower-noise TensorIR and fewer accidental temporaries reaching backend planning.
Caveats:
Primarily a cleanup pass; runtime wins are indirect.
Stage: [tensorize]

Optimization: [Map Chain Collapse]
Flag to enable: [--enable-opt map-chain-collapse]
Purpose:
Collapse nested elementwise TensorIR producers into a single elementwise body when shape and purity allow it.
Expected results:
Fewer temporary tensors and stronger pointwise fusion before backend planning.
Caveats:
Can increase scalar body complexity if used too aggressively.
Stage: [tensorize]

Optimization: [Reduce Map Fusion]
Flag to enable: [--enable-opt reduce-map-fusion]
Purpose:
Represent map-then-reduce pipelines as fused mapped reductions in TensorIR instead of deferring the decision to backend planning.
Expected results:
Lower temporary traffic and better reduction planning on dot, norm, and loss-style kernels.
Caveats:
Makes reduction nodes richer and relies on good reduction planning downstream.
Stage: [tensorize]

Optimization: [Small Producer Clone]
Flag to enable: [--enable-opt small-producer-clone]
Purpose:
Clone very small pure elementwise producers into consumers when that removes an intermediate tensor.
Expected results:
Fewer memory round-trips for tiny helper-produced intermediates.
Caveats:
Should stay thresholded to avoid code growth and over-fusion.
Stage: [tensorize]

Optimization: [Reduction Body Normalize]
Flag to enable: [--enable-opt reduction-body-normalize]
Purpose:
Canonicalize mapped reduction bodies and metadata so equivalent reduction pipelines share the same TensorIR form.
Expected results:
Better reduction matching and more stable planning for helper-shaped reductions.
Caveats:
Mostly an enabling pass; direct wins are indirect.
Stage: [tensorize]

Optimization: [Tensor Common Subexpression Elimination]
Flag to enable: [--enable-opt tensor-cse]
Purpose:
Merge structurally identical TensorIR nodes when they compute the same value.
Expected results:
Less duplicate work and fewer redundant intermediates reaching the planner.
Caveats:
Requires careful structural matching and is primarily useful on helper-expanded programs.
Stage: [tensorize]

Optimization: [Elementwise Fusion]
Flag to enable: [--enable-opt elementwise-fusion]
Purpose:
Fuse adjacent elementwise TensorIR nodes when producer and consumer are shape-compatible and single-use.
Expected results:
Fewer temporaries, fewer launches, and lower memory traffic for pointwise-heavy kernels.
Caveats:
Can increase kernel body complexity and register pressure; some tasks may regress.
Stage: [tensorize]

Optimization: [Scalar Hoist]
Flag to enable: [--enable-opt scalar-hoist]
Purpose:
Hoist tensor-lambda-invariant scalar lets out of tensor combinators before tensorization.
Expected results:
Simpler elementwise bodies, fewer duplicated scalar computations, and cleaner scalar capture sets.
Caveats:
Most helpful when frontend code contains local scalar helpers or repeated invariant lets.
Stage: [tensorize]

Optimization: [Materialization Choice]
Flag to enable: [--enable-opt materialization-choice]
Purpose:
Choose whether small tensor producers should be fused, cloned, or explicitly materialized from reuse and body-cost metadata.
Expected results:
Fewer harmful over-fusion cases on large pointwise pipelines and better mapped-reduction producer choices.
Caveats:
Heuristic and workload-sensitive; an overly conservative policy can miss good fusion wins.
Stage: [tensorize]

Optimization: [Reduction Accumulator Shape]
Flag to enable: [--enable-opt reduction-accumulator-shape]
Purpose:
Normalize mapped-reduction accumulator/update structure so helper-expanded reductions converge before planning.
Expected results:
More stable mapped-reduction TensorIR and better matching across expressive source forms.
Caveats:
Primarily an enabling pass; most wins come from downstream planning.
Stage: [tensorize]

Optimization: [Reduction Reuse Hoist]
Flag to enable: [--enable-opt reduction-reuse-hoist]
Purpose:
Hoist and simplify repeated scalar work inside mapped-reduction bodies before backend planning.
Expected results:
Smaller reduction bodies and better reuse for weighted and branchy mapped reductions.
Caveats:
Best paired with reduction canonicalization and map-reduce fusion.
Stage: [tensorize]

Optimization: [Reduction Body CSE]
Flag to enable: [--enable-opt reduction-body-cse]
Purpose:
Collapse repeated mapped-reduction subexpressions into simpler scalar forms before backend planning.
Expected results:
Lower scalar body cost for reduction kernels with repeated term structure and more stable mapped-reduction normalization.
Caveats:
Works through local scalar rewrites rather than full let-bound CSE, so it helps only when repeated structure is simple enough to fold.
Stage: [tensorize]

Optimization: [Reduction Materialization Choice]
Flag to enable: [--enable-opt reduction-materialization-choice]
Purpose:
Choose more carefully whether producers feeding reductions should be fused, cloned, or materialized from reduction-shape and body-cost metadata.
Expected results:
Fewer over-fused reduction inputs and better fixed-path behavior on weighted, ratio, and helper-expanded reductions.
Caveats:
Heuristic and workload-sensitive; too much conservatism can leave good fusion wins on the table.
Stage: [tensorize]

Optimization: [Reduction Late Fusion Guard]
Flag to enable: [--enable-opt reduction-late-fusion-guard]
Purpose:
Block reduction-directed fusion when the combined mapped body is too costly or branch-heavy for the current fixed planning heuristics.
Expected results:
Fewer large reduction regressions caused by overly aggressive map-into-reduce fusion.
Caveats:
Can suppress some profitable fusion when the complexity threshold is too low.
Stage: [tensorize]

Optimization: [Branch-Aware Fusion Guard]
Flag to enable: [--enable-opt branch-aware-fusion-guard]
Purpose:
Block aggressive fusion when branch-heavy producer bodies are likely to inflate large pointwise kernels.
Expected results:
Fewer branch-amplified regressions on workloads such as affine-clamp-like and HFT signal transforms.
Caveats:
Can disable useful fusion if the branchiness threshold is too low.
Stage: [tensorize]

Optimization: [Pointwise Materialization Guard]
Flag to enable: [--enable-opt pointwise-materialization-guard]
Purpose:
Prefer simpler materialized or lightly fused plans for expensive division-heavy or branch-heavy pointwise producers.
Expected results:
Better large-size behavior on previously regressing pointwise kernels.
Caveats:
Mostly useful on large shapes; neutral or slightly negative on tiny elementwise kernels.
Stage: [tensorize]

Optimization: [Reduction Plan Specialization]
Flag to enable: [--enable-opt reduction-plan-specialize]
Purpose:
Pick stronger fixed launch heuristics for reduction kernels during TensorIR to KernelPlan lowering.
Expected results:
Large fixed-path improvements on reduction-heavy workloads such as dot, norms, and loss reductions.
Caveats:
Overlaps with Triton autotune benefits and may not help very small reductions.
Stage: [backend]

Optimization: [Elementwise Plan Specialization]
Flag to enable: [--enable-opt elementwise-plan-specialize]
Purpose:
Pick stronger fixed launch heuristics for elementwise kernels based on input count and body complexity.
Expected results:
Incremental non-autotuned wins for simple pointwise kernels and better starting points for autotune.
Caveats:
Usually a smaller gain than fusion; can be neutral on already-well-matched kernels.
Stage: [backend]

Optimization: [Reduction Precombine]
Flag to enable: [--enable-opt reduction-precombine]
Purpose:
Fuse a single-use elementwise producer into the first reduction stage during KernelPlan lowering.
Expected results:
Eliminates an intermediate tensor and reduces launch/memory overhead for map-then-reduce pipelines.
Caveats:
Only applies when the mapped producer is single-use and body complexity stays under a guard threshold.
Stage: [backend]

Optimization: [Reduction Two Phase]
Flag to enable: [--enable-opt reduction-two-phase]
Purpose:
Switch small reductions to a direct single-block path while retaining recursive multi-pass reduction for larger inputs.
Expected results:
Lower workspace pressure and better small/medium reduction latency.
Caveats:
Threshold-sensitive; gains depend on workload size distribution.
Stage: [backend]

Optimization: [Launch Bucket Specialization]
Flag to enable: [--enable-opt launch-bucket-specialize]
Purpose:
Choose fixed launch settings from size buckets instead of one global default.
Expected results:
Better fixed-path performance across mixed small, medium, and large workloads.
Caveats:
Adds more fixed-policy surface and needs good bucket defaults to avoid regressions.
Stage: [backend]

Optimization: [Reduction Classification]
Flag to enable: [--enable-opt reduction-classify]
Purpose:
Classify reductions by source form and scalar body complexity before kernel planning.
Expected results:
Better generic planning decisions for plain, mapped, branchy, and weighted reductions.
Caveats:
Mainly an enabling pass; planning wins depend on downstream heuristics.
Stage: [backend]

Optimization: [Reduction Tree Planning]
Flag to enable: [--enable-opt reduction-tree-plan]
Purpose:
Choose stronger generic reduction tree shapes from reduction class and size bucket.
Expected results:
Lower steady-state latency on large reductions without backend-specific tuning logic.
Caveats:
Heuristic-driven and may need per-workload calibration.
Stage: [backend]

Optimization: [Reduction Split Planning]
Flag to enable: [--enable-opt reduction-split-plan]
Purpose:
Choose direct, single-stage, or multi-stage reduction plans from size and reduction class.
Expected results:
Better small/medium/large reduction behavior with clearer planning metadata.
Caveats:
Interacts with single-block thresholds and reduction tree choices.
Stage: [backend]

Optimization: [Reduction Stage Sizing]
Flag to enable: [--enable-opt reduction-stage-sizing]
Purpose:
Choose reduction stage counts from reduction class, body cost, and configurable size thresholds rather than one coarse split.
Expected results:
More stable large-reduction performance and fewer mis-sized fixed reduction plans.
Caveats:
Threshold-heavy and may overlap with other reduction planning passes.
Stage: [backend]

Optimization: [Reduction Stage Balance]
Flag to enable: [--enable-opt reduction-stage-balance]
Purpose:
Balance fixed reduction launch shape and stage layout from reduction class, body cost, and repeated-subexpression pressure.
Expected results:
Better medium/large fixed-path reductions without relying on backend autotune.
Caveats:
Still heuristic-driven and may need retuning as the reduction suite evolves.
Stage: [backend]

Optimization: [Reduction Partial Shape Pack]
Flag to enable: [--enable-opt reduction-partial-shape-pack]
Purpose:
Choose reduction launch shapes that pack intermediate partial buffers more tightly for multi-stage reductions.
Expected results:
Lower partial-buffer footprint and better fixed-path reduction latency on large inputs.
Caveats:
Mostly helps staged reductions; plain small reductions may see no change.
Stage: [backend]

Optimization: [Reduction Class-Aware Buckets]
Flag to enable: [--enable-opt reduction-class-aware-buckets]
Purpose:
Adjust fixed reduction launch buckets by mapped-reduction class instead of reusing one bucket table for every reduction shape.
Expected results:
Better fixed launch choices across plain, weighted, branchy, and ratio-style reductions.
Caveats:
Adds more policy surface and needs conservative defaults to avoid new regressions.
Stage: [backend]

Optimization: [Temporary Lifetime Packing]
Flag to enable: [--enable-opt temp-lifetime-pack]
Purpose:
Reuse temporary storage slots when plan step lifetimes do not overlap.
Expected results:
Lower temporary count and less intermediate buffer pressure.
Caveats:
Requires correct liveness modeling; mostly affects large pipelines.
Stage: [backend]

Optimization: [Storage Reuse Packing]
Flag to enable: [--enable-opt storage-reuse-pack]
Purpose:
Turn temporary lifetime analysis into real storage-name reuse inside KernelPlan rather than metadata-only slot annotations.
Expected results:
Lower intermediate buffer count and more faithful temporary reuse in generated backends.
Caveats:
Requires careful input/output renaming so reused storage never clobbers live values.
Stage: [backend]

Optimization: [Multi-Use Fusion Guard]
Flag to enable: [--enable-opt multi-use-fusion-guard]
Purpose:
Prevent planning decisions that over-fuse producers with reuse patterns likely to regress.
Expected results:
More robust performance on workloads where aggressive fusion inflates body complexity or duplicate work.
Caveats:
Can disable some wins if the guard is too conservative.
Stage: [backend]

Optimization: [Pointwise Shape Planning]
Flag to enable: [--enable-opt pointwise-shape-plan]
Purpose:
Choose better generic pointwise plan parameters from body complexity and shape bucket.
Expected results:
Stronger fixed-path behavior for large pointwise kernels without backend-specific scheduling.
Caveats:
Incremental on already well-matched small kernels.
Stage: [backend]

Optimization: [Shared Body Trait Analysis]
Flag to enable: [--enable-opt shared-body-trait-analysis]
Purpose:
Classify TensorIR scalar bodies into backend-neutral traits before target
planning.
Expected results:
Keeps Triton and CUDA pattern recognition coherent while allowing each backend
to lower traits into different launch and instruction choices.
Caveats:
Diagnostic by itself; runtime changes require backend-specific instruction
selection flags.
Stage: [backend]

Optimization: [Triton Reduction Instruction Selection]
Flag to enable: [--enable-opt triton-reduction-instruction-select]
Purpose:
Use shared reduction traits to specialize fixed-path generated Triton reduction
families and launch buckets.
Expected results:
Lower small and medium generated Triton reduction latency, especially for
dot-product, weighted-product, norm-square, robust, branchy, and ratio bodies.
Caveats:
Triton-only; should be benchmarked against fixed external Triton and CUDA
baselines because small reductions are launch-policy sensitive.
The backend keeps tiny and small specialized reductions such as dot-product,
norm-square, weighted-product, and delta-square on the parallel small-partial
path instead of capping them to a few strided programs; this preserves the
specialized family labels without serializing the first reduction stage.
For fixed-path specialized reductions, backend codegen also keeps launch buckets
at least as wide as the reduction default block size, matching the high-throughput
Triton reduction path instead of selecting a narrower medium bucket.
Stage: [backend]

Optimization: [Triton Pointwise Instruction Selection]
Flag to enable: [--enable-opt triton-pointwise-instruction-select]
Purpose:
Use shared pointwise traits to specialize fixed-path generated Triton pointwise
launch buckets for activation, clamp, and filter-like bodies.
Expected results:
Better generated Triton fixed-path behavior on small pointwise kernels without
requiring autotune.
Caveats:
Triton-only; broad pointwise launch changes can regress memory-bound large
kernels if not guarded by traits.
The Triton code generator lowers direct min/max and select-shaped min/max
patterns to `tl.minimum`/`tl.maximum` instead of generic `tl.where` when the
pattern is semantics-preserving.
Stage: [backend]

Optimization: [Triton Small Reduction Tail Tuning]
Flag to enable: [--enable-opt triton-small-reduction-tail-tune]
Purpose:
Apply a narrower fixed-path launch policy for trait-classified small Triton
reductions that still lag handwritten fixed Triton kernels.
Expected results:
Lower tail latency for small dot, norm, weighted, and loss-style reductions
without changing large-size Triton behavior.
Caveats:
Triton-only and intentionally size-sensitive; accept only after focused and
full-suite regression checks.
Stage: [backend]

Optimization: [CUDA Pointwise Vectorization]
Flag to enable: [--enable-opt cuda-pointwise-vectorize]
Purpose:
Route eligible CUDA pointwise plans through a CUDA-native vectorized `float4`
kernel with a scalar fallback path.
Expected results:
Lower memory-traffic overhead on eligible pointwise kernels while retaining the
scalar path for small sizes and tail elements.
Caveats:
CUDA-only; intentionally rejected by the Triton backend. The planner keeps this
guarded to alignment-safe non-ratio pointwise bodies because vectorizing
expensive division-heavy bodies has not been consistently profitable.
The CUDA lowering also maps clamp and identity-vs-threshold select bodies to
`fminf`/`fmaxf` so vectorized activation and affine-clamp kernels do not emit
unnecessary ternary instruction sequences.
Stage: [backend]

Optimization: [CUDA Pointwise Tail Tuning]
Flag to enable: [--enable-opt cuda-pointwise-tail-tune]
Purpose:
Use shared pointwise body traits to select CUDA-specific scalar, small-N, and
vectorized pointwise families for the remaining branch, threshold, and ratio
tails.
Expected results:
Close small and medium pointwise gaps without changing the shared KernelPlan
architecture or Triton behavior. For branch-heavy order-book filter bodies and
mixed filter/ratio bodies, the CUDA wrapper may keep the vectorized family for
small and medium sizes but fall back to scalar grid-stride code for large inputs
when vector lane duplication is likely to increase register pressure. Large
threshold-only activation bodies may use the same scalar fallback when the
branchy vector body is less stable.
Caveats:
CUDA-only and intentionally trait-sensitive; accept only after focused pass
comparisons against the previous fixed CUDA profile.
Stage: [backend]

Optimization: [CUDA Selected Pointwise Tail Planning]
Flag to enable: [--enable-opt cuda-pointwise-selected-tail-plan]
Purpose:
Apply CUDA-specific, trait-selected expression and launch choices for pointwise
bodies whose best generated plans differ from the shared generic policy.
Expected results:
Reduce remaining affine vector update pointwise gaps without changing Triton or
reduction behavior. Selected CUDA affine-vector kernels may vectorize eagerly
instead of routing medium-sized bandwidth-bound maps through the scalar tail
path first, while small affine-vector update tails can keep the scalar one-pass
route when launch/occupancy effects dominate. Selected activation/vector-update
kernels may use a guarded `float4` loop shape matching the handwritten CUDA
baselines. Very large
selected activation/vector-update kernels may fall back to scalar grid-stride
code when that path is more stable on bandwidth-bound tails. Selected
affine-vector wrappers also elide unreachable lower-bound scalar routing after
the public C ABI has already rejected non-positive sizes. Non-branching
ratio-book pointwise bodies may lower scalar division through CUDA's fast
single-precision divide intrinsic when the selected CUDA plan is active.
Selected one-input affine-clamp wrappers may also use the guarded vector path
immediately for small benchmark sizes instead of detouring through the scalar
single-pass launch, and the selected vector path can emit a single-pass vector
kernel when the vector tile count fits the configured pointwise launch cap.
Simple activation wrappers keep the scalar one-pass route
for small sizes while still vectorizing medium sizes. Low-complexity
multiply-bearing ratio-book pointwise bodies may use the same immediate vector
path for small and medium sizes while falling back to scalar code for large
tails where the scalar route has been more stable. Simpler non-multiply
ratio-book, non-ratio filter/book, and mixed filter/ratio bodies keep the
scalar/generic route because the extra vector lane work has not paid off there.
Caveats:
CUDA-only and intentionally narrow; keep a same-run former-self baseline and
non-held-out regression guard for every selected trait family.
Stage: [backend]

Optimization: [CUDA Medium Pointwise Planning]
Flag to enable: [--enable-opt cuda-pointwise-medium-plan]
Purpose:
Route selected medium-size generic branchy threshold/clip CUDA pointwise bodies
through scalar loop codegen with a handwritten-style block cap.
Expected results:
Close remaining medium-size affine clamp and soft-threshold gaps without
disturbing simple activation vector wins. One-input affine clamp bodies can use
the vectorized CUDA pointwise family for the medium range while preserving
scalar fallback for small and large ranges.
Caveats:
CUDA-only and size-sensitive; accept only after non-held-out regression gates
pass against the previous CUDA fixed profile.
Stage: [backend]

Optimization: [CUDA Predicated Select Lowering]
Flag to enable: [--enable-opt cuda-pointwise-predicated-select]
Purpose:
Lower clamp and threshold-shaped CUDA scalar bodies to math/select idioms such
as `fminf`, `fmaxf`, and `copysignf` instead of branchy per-element updates.
Expected results:
Reduce branch overhead for activation, clamp, soft-threshold, and similar
pointwise kernels in scalar and vectorized generated CUDA code.
Caveats:
Pattern-based by design; mixed branch-heavy code keeps its existing lowering
unless the body matches a recognized safe select shape. The CUDA selected tail
profile may keep affine clip/threshold pointwise bodies in explicit branch form
when focused data shows that shape matching handwritten CUDA better than
`fminf`/`fmaxf` lowering.
Stage: [backend]

Optimization: [CUDA Affine Vector Tail Tuning]
Flag to enable: [--enable-opt cuda-pointwise-affine-vector-tail]
Purpose:
Use CUDA-specific vector/tail routing for affine vector updates while leaving
ratio and branch-heavy pointwise bodies on their existing paths.
Expected results:
Reduce small and medium AXPY-style pointwise overhead without changing
reduction planning or ratio-book codegen.
Caveats:
Applies only to trait-classified affine vector updates and must be validated
against the non-held-out CUDA regression gate before promotion.
Stage: [backend]

Optimization: [CUDA Book Filter Register Planning]
Flag to enable: [--enable-opt cuda-book-filter-register-plan]
Purpose:
Keep complex order-book/filter pointwise bodies on scalar codegen when vector
lane duplication is likely to increase register pressure.
Expected results:
Improve large branch-heavy HFT pointwise tails while preserving simpler
vectorized pointwise wins.
Caveats:
High-risk and intentionally optional; enable only when focused benchmark data
show a clear non-held-out win.
Stage: [backend]

Optimization: [CUDA Reduction Shuffle]
Flag to enable: [--enable-opt cuda-reduction-shuffle]
Purpose:
Use warp/block shuffle helpers for selected CUDA sum reductions instead of the
existing shared-memory tree.
Expected results:
Lower reduction overhead for simple mapped, dot-product, norm-square, and
delta-square reductions.
Caveats:
CUDA-only; intentionally rejected by the Triton backend. Weighted-product and
branchy/ratio reductions currently stay on the shared-memory path because the
shuffle path has regressed those generated plans in focused CUDA measurements.
Stage: [backend]

Optimization: [CUDA Reduction Tail Tuning]
Flag to enable: [--enable-opt cuda-reduction-tail-tune]
Purpose:
Route selected branchy, robust, and ratio-style CUDA reductions through the
existing atomic-output path when workspace/combine overhead dominates tail
latency.
Expected results:
Reduce the remaining medium-size CUDA reduction tail without changing the CUDA
backend architecture or public artifact contract.
Caveats:
CUDA-only and atomic-accumulation sensitive; accept only after focused and
full-suite regression checks.
Stage: [backend]

Optimization: [CUDA Medium Reduction Tail Planning]
Flag to enable: [--enable-opt cuda-reduction-medium-tail-plan]
Purpose:
Extend CUDA reduction tail routing for medium branchy reductions whose
workspace/combine overhead dominates fixed-path latency.
Expected results:
Reduce remaining medium-size robust and branchy reduction gaps without changing
the CUDA reduction architecture. The current checks also cover clipped robust
reductions whose workspace tree still dominates small-size latency.
Caveats:
Atomic-output routing is workload-sensitive; promote only after focused and
full non-held-out regression gates pass.
Stage: [backend]

Optimization: [CUDA Async Wrapper Return]
Flag to enable: [--enable-opt cuda-async-wrapper-return]
Purpose:
Return from generated CUDA wrappers after launch-error checks and rely on the
caller-side synchronization boundary used by the runtime harness.
Expected results:
Avoid redundant per-wrapper device synchronization in CUDA runtime measurements
and in callers that already synchronize at a higher level.
Caveats:
CUDA-only; intentionally rejected by the Triton backend. Host callers that need
synchronous completion must synchronize after calling the generated wrapper.
Stage: [backend]
