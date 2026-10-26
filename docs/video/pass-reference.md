# Deuteron pass reference

This reference follows the exact names and order of the [27-entry README inventory](../../README.md#compiler-passes). [passes.json](passes.json) supplies the same entries for the optional reference slides. The inventory includes analyses, structural IR rewrites, policies, and specialized lowering routines; it is not a claim of 27 independently scheduled passes.

The input is already scheduled Emmy Kernel IR. These mechanisms inspect its structure and select a supported rewrite or lowering. Pattern-specific checks are not a general memory-safety proof. A declined optimization keeps the existing lowering; malformed or unsupported input may still produce an error.

The initial rewrite order in [build.py](../../compiler/build.py) is elementwise vectorization → reduction load forwarding → recurrence-axis reordering → split-K grid folding. [Lowering.__init__](../../compiler/codegen/lower.py) then applies eligible hybrid normalization, packed register staging, and small-tree elimination. The function [run_passes](../../compiler/passes/__init__.py) returns contraction and recurrence plans for specialized emitters; it is not the entire pipeline.

Before/after examples below are schematic transformations, not new benchmarks or measured ablations. FP32 reductions may change association, and tensor/approximate-math modes have the precision limits stated in the README. None of these entries claims a standalone speedup or first-ever algorithm.

## P01. Symbolic index and effect analysis

**Kind:** Analysis helper, not a standalone IR rewrite. **Source:** [compiler/analysis.py](../../compiler/analysis.py); `canonical`, `difference`, `constant_difference`, `known_multiple`, `uses`, `exprs`.

**Before → after:** `2*i + i + 4` and `3*i + 4` become the same coefficient map for comparison. This supplies a proof to another transformation; it does not itself rewrite the emitted address.

**Key checks:** Integer coordinates and supported operators; non-affine products decline.

Selects, division, remainder, and some bit operations can remain opaque structural terms. Pass-specific bounds checks are still required; this is not a general symbolic solver.

## P02. Storage typing

**Kind:** Type and eligibility analysis, used before rewriting and during lowering. **Source:** [compiler/storage.py](../../compiler/storage.py); `half_storage_types`; conversions in `Lowering.convert`.

**Before → after:** Half-buffer values used by an explicit FP32 operation are widened before arithmetic; a half output is narrowed at the store. A raw half copy remains a bit-preserving copy.

**Key checks:** f16/f32 global buffers, control-only indices, explicit f32 results, plain stores.

The supported half subset excludes shared-memory/tensor paths and half atomics. BF16 and TF32 tensor operands are internal formats, not public BF16/TF32 buffer support.

## P03. Elementwise vectorization

**Kind:** Kernel IR rewrite. **Source:** [compiler/passes/vectorize.py](../../compiler/passes/vectorize.py); `vectorize_elementwise`, `_match`, `_rewrite`.

**Before → after:** `for i=t; i<N; i+=B: y[i]=f(x[i])` becomes a loop over `q=t; q<N/4; q+=B` with four-element loads/stores at `4*q` and four corresponding scalar operations.

**Key checks:** Extent divisible by 4×threads; contiguous indices; safe overlap and lifetimes.

B is the block width. Stores must be ordinary and unswizzled. Same-buffer reads/writes must use matching indices; temporary values cannot escape the rewritten loop. The structural width here is four.

## P04. Reduction load forwarding

**Kind:** Kernel IR rewrite. **Source:** [compiler/passes/regforward.py](../../compiler/passes/regforward.py); `forward_reduction_loads_register`, `_forward_pair`.

**Before → after:** `sum(x*x); reread x; y=x*scale` becomes `load and retain x; sum(x*x); y=retained_x*scale`. The reduction and its later consumer share the loaded values.

**Key checks:** Readonly matching input, no writes, intervening CTA barrier, ≤8 vector trips.

The consumer has matching four-wide indices and at most eight vector iterations per thread. Scalar lifetimes and generated names must be closed. Reduction association can change; this is not a universal bitwise-equivalence claim.

## P05. Recurrence-axis reordering

**Kind:** Kernel IR rewrite. **Source:** [compiler/passes/recurrence.py](../../compiler/passes/recurrence.py); `reorder_recurrences`.

**Before → after:** For a score depending on row and head but not output column, logical axes `(row, column, head)` can become `(row, head, column)`. Adjacent lanes can then share the same score work.

**Key checks:** Closed named coordinates, exact tile coverage, no rasterization, wider sharing.

The example illustrates the ordering rule, not an unconditional match. A newly recognized recurrence must have a strictly larger sharing width; physical launch dimensions and logical output coordinates remain unchanged.

## P06. Split-K grid folding

**Kind:** Kernel IR rewrite. **Source:** [compiler/passes/gridfold.py](../../compiler/passes/gridfold.py); `fold_grid_contractions`, `_fold`.

**Before → after:** `S` logical split-K partitions of depth `D` become `S/f` partitions of depth `f*D`. Each active CTA accumulates a larger partial and still performs the original atomic output update.

**Key checks:** Atomic dense contraction; affine coverage; depth≤32, shared≤48 KiB, enough work.

This merges existing split-K partitions; it does not introduce split-K parallelism or eliminate atomics. The factor is a power of two, divides S, leaves at least 32,768 scheduled threads, and fits the static shared budget. The original physical grid remains; surplus CTAs exit through the tile guard.

## P07. Cooperative dot reduction

**Kind:** Recognition helper plus specialized loop lowering. **Source:** [compiler/transforms.py](../../compiler/transforms.py); `sharing_width`; `Lowering.loop`.

**Before → after:** Instead of every lane evaluating the same complete dot product, lane `l` sums `k=l mod W, l mod W+W, ...`; butterfly shuffles combine the W partial sums.

**Key checks:** Closed load/load/product/sum loop, full warps, uniform scope, width≤32.

The dot length must be at least W and the participating output axes must prove input reuse. Divergent lowering disables this cooperative path. Addition order changes.

## P08. Coalesced sum/square reduction

**Kind:** Recognition helper plus specialized loop lowering. **Source:** [compiler/transforms.py](../../compiler/transforms.py); `striped_reduction`; `Lowering.loop`, `Lowering.register_reduction`.

**Before → after:** A scalar `i=t; i+=B` sum becomes contiguous four/eight-value loads with a `width*B` stride. Eligible long loops use 32 FP32 accumulators and alternating loaded batches before the final sum.

**Key checks:** Seeded sum/square with matching warp/tree reduction; long path also readonly.

The long path requires add/multiply operations, readonly global storage, no writes to that input, no explicit unroll option, at least 32 vector trips, and divisibility by `64/width`. Emitting lookahead loads does not guarantee a particular final hardware schedule.

## P09. Small-tree elimination

**Kind:** Kernel IR rewrite. **Source:** [compiler/passes/tailtree.py](../../compiler/passes/tailtree.py); `linearize_small_trees`, `_scalar_tree`.

**Before → after:** For four scalar partials, repeated shared tree steps become loads of `p0..p3` and a balanced expression `(p0+p1)+(p2+p3)`. Eligible multiple-state combines use a subgroup shuffle instead.

**Key checks:** 2–8 power-of-two partials, producer barrier, exclusive scratch, closed states.

The producer CTA barrier remains. Scratch must have no later users, and cross-boundary memory effects must permit removing the tree phases. The scalar tree can use a different FP32 association.

## P10. Warp tree specialization

**Kind:** Specialized collective lowering. **Source:** [compiler/codegen/lower.py](../../compiler/codegen/lower.py); `Lowering.tree`.

**Before → after:** A multi-step shared-memory halving tree becomes warp-0 shared loads and shuffles, one shared result store, one CTA barrier, and a result load by all threads.

**Key checks:** Enabled specialization, warp-indexed tree, no inner dimension, length≤32.

Other trees retain the generic lowering. Existing producer synchronization is not removed by this mechanism.

## P11. Contraction tensorization

**Kind:** Plan recognizer plus tensor-instruction lowering. **Source:** [compiler/passes/contraction.py](../../compiler/passes/contraction.py); `tensorize`, `_contraction`; `TensorEmitter.mma`.

**Before → after:** Shared FP32 multiply/add loops become BF16x3 or TF32x3 MMA: low×high, high×low, and high×high products accumulate into FP32.

**Key checks:** Complete affine tiles, full warps, closed scalar scope, supported M/N/K sizes.

The low×low term is omitted, so these approximate FP32 multiplication. Tile rows must be a multiple of 16, columns of 8, and K compatible with the selected 16- or 8-wide MMA. The f32 mode keeps ordinary FP32 products.

## P12. Tensor shared-memory layout

**Kind:** Tensor address and memory-operation lowering. **Source:** [compiler/codegen/tensor.py](../../compiler/codegen/tensor.py); `TensorEmitter.remap`, `packed_address`, `packed_store`, `ldmatrix`; `shared_layout`.

**Before → after:** Plain shared row/column addresses become matching XOR-remapped producer/consumer addresses. Eligible packed BF16 fragments use `ldmatrix` instead of scalar fragment assembly.

**Key checks:** Recognized tensor operands; compatible dimensions and packed BF16 eligibility.

The remapping is internal to recognized operand buffers, not a general user-visible shared-memory swizzle feature.

## P13. Packed register staging

**Kind:** Kernel IR rewrite cooperating with packed tensor lowering. **Source:** [compiler/passes/regstage.py](../../compiler/passes/regstage.py); `stage_register_loads`, `_stage`; `TensorEmitter.packed_store`.

**Before → after:** A multi-stage FP32 asynchronous matrix-copy ring becomes vector global lookahead loads in registers and one shared stage containing BF16 high/residual components.

**Key checks:** Exact original future phase, complete aligned copies, readonly inputs, closed ring.

The proof checks original clamped future source coordinates before replacing them, as well as prologue phases, bounds, coverage, alignment, names, and scalar/buffer ownership. The public wrong-prefetch test rejects both a bare current-coordinate replacement and a canceled future expression.

## P14. Tensor fragment pipelining

**Kind:** Tensor loop lowering. **Source:** [compiler/codegen/tensor.py](../../compiler/codegen/tensor.py); `TensorEmitter._emit_packed`; outer-loop policy in `Lowering.loop`.

**Before → after:** Repeated fragment-load/MMA blocks become a compact loop with current fragments in phi nodes: load following fragments, consume current fragments, carry the following set into the next iteration.

**Key checks:** Certified packed BF16 operands; compatible fragment depth and loop bounds.

The fragment loop disables unrolling. The surrounding packed K loop defaults to unroll two when no explicit unroll is supplied. The final fragment is consumed in a separate tail.

## P15. Pipeline drain

**Kind:** Part of the packed register-staging rewrite. **Source:** [compiler/passes/regstage.py](../../compiler/passes/regstage.py); `_stage` in regstage.py: `pending`, `prefetch`, and `drain` construction.

**Before → after:** An unconditional next-tile prefetch and post-consumer barrier become conditional on `current + depth < extent`. The final iteration needs neither.

**Key checks:** Certified staging loop with uniform next-tile predicate and no later shared users.

Current-tile producer stores and their producer barrier remain. This is a phase change inside P13, not a separately scheduled compiler pass.

## P16. Staged online recurrence

**Kind:** Recurrence plan recognizer plus specialized emitter. **Source:** [compiler/passes/recurrence.py](../../compiler/passes/recurrence.py); `stage_recurrences`, `_recurrence`, `_stage_operands`; `RecurrenceEmitter.emit`.

**Before → after:** An online max/mass/weighted-sum recurrence becomes score computation, maximum reduction, shared probability storage, mass reduction, and value accumulation.

**Key checks:** Exact recurrence algebra, readonly operands, uniform scale, bounded rounds/scratch.

Each computed unnormalized weight directly updates register mass and is also stored for PV; mass does not reread shared weights. The recognizer checks the recurrence and output-normalization dataflow. It may stage fully covered reusable operands with cp.async when traffic justifies it and the static shared budget permits. It does not fuse an arbitrary unscheduled attention graph.

## P17. Blocked recurrence loads and address hoisting

**Kind:** Address/load policy inside recurrence lowering. **Source:** [compiler/passes/recurrence.py](../../compiler/passes/recurrence.py); `RecurrenceEmitter.emit`: `packed`, `keys`, and `bases`.

**Before → after:** Key indices `lane + r*W` can become `lane*rounds + r` to expose adjacent loads. Recomputed value addresses become one base pointer plus `key*stride`.

**Key checks:** Compatible 2/4 key rounds; unstaged operands; affine bounded full value coverage.

Key packing and address hoisting have separate guards. Hoisting requires one key-axis coordinate, no hidden key dependence in other coordinates, affine expressions, and a buffer smaller than 2³¹ elements.

## P18. Warp-distributed vector PV reduction

**Kind:** Guarded specialized recurrence lowering. **Source:** [compiler/passes/recurrence.py](../../compiler/passes/recurrence.py); `_grouped_value_axis`; `RecurrenceEmitter.grouped_values`.

**Before → after:** One scalar output column per lane becomes four groups of eight lanes: keys are `4*i + lane//8`, and each lane holds columns `4*(lane%8) + 0..3`. Shuffles at distances 16 and 8 combine groups.

**Key checks:** Single full warp, 32 columns, one component, full ownership, replayable suffix.

The first eight lanes write four columns each. Before replaying the normalization suffix for each column, the saved denominator is restored, including when the suffix overwrites that denominator in place.

## P19. Recurrence shared-storage reuse

**Kind:** Lifetime analysis plus allocation alias selection. **Source:** [compiler/passes/lifetime.py](../../compiler/passes/lifetime.py); `recurrence_storage_is_warp_private`; `RecurrenceEmitter.__init__`.

**Before → after:** Separate staged-invariant and probability buffers can share storage after the invariant reads are dead.

**Key checks:** Same warp/component ownership; finite negative identity; bounded full mapping.

Every conflicting read must belong to the same warp and recurrence component as the overwrite. Read→maximum→probability-store data dependence supplies the ordering; the proof does not treat a shuffle as a generic memory fence. Otherwise storage stays separate.

## P20. Hybrid row normalizer

**Kind:** Kernel IR rewrite with a specialized maximum hint. **Source:** [compiler/passes/hybrid_normalizer.py](../../compiler/passes/hybrid_normalizer.py); `hybrid_normalizer`, `_hybrid`, `_normalizer`.

**Before → after:** A coupled max/mass shared tree plus a rereading consumer becomes a row maximum within each warp, exponentials of each thread’s owned cached values, and only scalar mass communicated across warps.

**Key checks:** Closed max/mass tree, finite seed, 64–256 threads, bounded rows, approximate math.

This recognizes the complete normalizer/consumer shape, not an arbitrary maximum reduction. Supported threads are powers of two; values per thread are 1, 2, or 4 and the row work is bounded.

## P21. Ordered hardware warp maximum

**Kind:** Shared hardware-reduction lowering helper. **Source:** [compiler/codegen/lower.py](../../compiler/codegen/lower.py); `Lowering.ordered_maximum`; callers `Lowering.shuffle`, `RecurrenceEmitter.reduce`.

**Before → after:** Butterfly FP32 maximum steps become float-bit ordering, `redux.sync.max.s32`, and reversal of the bit mapping.

**Key checks:** Full warp; non-NaN identity fold; recurrence CTA must have more than 32 threads.

The callers establish the non-NaN maximum inputs. Certified hybrid normalizers and multi-warp recurrences use it; single-warp recurrences retain shuffles. This is not a hardware FP32 sum/redux claim.

## P22. CTA/thread address factoring

**Kind:** Tile address lowering. **Source:** [compiler/codegen/lower.py](../../compiler/codegen/lower.py); `Lowering.tile`.

**Before → after:** For axes `(tile, row, col)` with `rows*cols=B`, decoding `g=ctaid*B+tid` becomes `tile=ctaid`, `row=tid//cols`, and `col=tid%cols`.

**Key checks:** Trailing-axis product equals block width; no rasterization; bounded tile ownership.

Partial logical grids retain an ownership guard. The optimization avoids repeated full-ticket arithmetic; it does not change the supplied grid or block.

## P23. Loop cost policy

**Kind:** Compile-time policy and loop metadata, not an autotuner. **Source:** [compiler/transforms.py](../../compiler/transforms.py); `default_unroll`; `Lowering.loop`; `RecurrenceEmitter.loop_n`.

**Before → after:** An ordinary loop receives a structure-sensitive unroll count or disable hint; eligible reductions additionally receive vector-width/interleave metadata.

**Key checks:** Recognized loop structure; explicit options; specialized coalesced-loop policy.

Unroll values are hints to LLVM, not a guarantee. Explicit unroll bypasses the automatic long-register pipeline, but the coalesced path can raise the count to its trip count when there are at most 128 trips. The packed fragment loop remains explicitly compact.

## P24. Reciprocal-square-root folding

**Kind:** Local dataflow recognition during expression lowering. **Source:** [compiler/transforms.py](../../compiler/transforms.py); `reciprocal_sqrt_source`; `Lowering.body`.

**Before → after:** `t=sqrt(x); y=reciprocal(t)` lowers y directly through `rsqrt.approx.ftz.f32(x)` when the optional approximation is enabled.

**Key checks:** Adjacent matching assignments, no self-overwrite, approximate math enabled.

If t has another use, its computation can remain. This is an approximate-math choice; disabling it does not make all other math universally IEEE-exact because exponentiation still uses approximate GPU instructions.

## P25. Dead atomic-result elimination

**Kind:** Post-emission PTX text rewrite. **Source:** [compiler/ptx.py](../../compiler/ptx.py); `rewrite_dead_atom_to_red`, `_register_used`.

**Before → after:** `atom...global.add...f32 result,[address],value` becomes `red...global.add...f32 [address],value` when the returned result is unused.

**Key checks:** Recognized FP32 global-add form and no later destination-register use.

The memory update remains. Recognized predicate, order, scope, and noftz qualifiers are retained; unrecognized forms are left untouched.

## P26. Fixed block contract

**Kind:** Post-emission PTX contract rewrite. **Source:** [compiler/ptx.py](../../compiler/ptx.py); `replace_maxntid_with_reqntid`.

**Before → after:** `.maxntid x[,y,z]` becomes `.reqntid x[,y,z]`, telling the assembler that the supplied block dimensions are required.

**Key checks:** Recognized maxntid directive and the compiler’s fixed-launch contract.

The caller must retain those launch dimensions. This is a launch contract, not a new kernel scheduling search.

## P27. Approximate division cleanup

**Kind:** Post-emission PTX text rewrite. **Source:** [compiler/ptx.py](../../compiler/ptx.py); `rewrite_div_rn_to_approx`; composed by `rewrite_ptx`.

**Before → after:** `div.rn.f32` becomes `div.approx.f32`. With approximate math disabled, that rewrite leaves the emitted division unchanged.

**Key checks:** approximate_math enabled and an exact div.rn.f32 opcode match.

This is a precision policy, not an IEEE-equivalence claim.

## LLVM stages and policy

LLVM supplies optimization and NVPTX code generation separately from the 27 mechanisms above. The exact driver is [build.py, _compile_uncached](../../compiler/build.py).

1. Verify the generated LLVM IR.
2. Run an explicit **SROA** pass to promote eligible temporary scalar storage.
3. Run the target-aware **LLVM O0–O3 module pipeline** selected by the optimization level. Loop vectorization is enabled when `vector_width > 1`; SLP vectorization is enabled at O2/O3.
4. At nonzero optimization levels, run cleanup: **Loop Strength Reduction** only when `lowering.striped` was set, then **NewGVN**, **InstCombine**, and **SimplifyCFG**.
5. Verify the optimized module, run **NVPTX instruction selection/code generation**, then apply P25–P27.

These are the explicitly requested LLVM stages; the standard O0–O3 pipeline contains additional LLVM-owned internal passes. Deuteron does not implement or claim authorship of those LLVM algorithms.

[Lowering.intrinsic](../../compiler/codegen/lower.py) can mark `activemask` and butterfly-shuffle intrinsics `willreturn`. Ordinary arithmetic permits FMA contraction; selected reduction paths additionally permit reassociation and signed-zero simplification. Loop hints do not guarantee final instruction counts or scheduling. The current target policy emits PTX 8.0 below SM100, PTX 8.7 for SM100/SM120, and PTX 9.0 for SM110; assembler and hardware support remain separate requirements.

## Source and test evidence

- [Tensor/copy tests](../../tests/test_tensor_copy.py) check BF16x3/TF32x3 instructions, the F32 cp.async path, assembly, all three precision modes, and rejection of two semantically wrong prefetch-phase forms (`test_wrong_prefetch_phase_declines_register_packing`).
- [Attention tests](../../tests/test_attention.py) check complete PV key/column ownership, emitted vector loads and shuffles, rejected control/output/suffix cases, and an in-place denominator suffix. The CUDA test varies key counts and checks both suffix forms.
- [FP16 tests](../../tests/test_f16_storage.py) check storage guards, half bit patterns, widening, narrowing boundaries, explicit FP32 intermediates, and unaligned accesses. `test_strided_loop_vectorization_preserves_half_storage` checks a four-half LLVM load and GPU output.
- [Interface tests](../../tests/test_interface.py) cover GPU-free target emission and a partial final block; [validation tests](../../tests/test_validation.py) cover malformed launch/options/IR cases. These are distinct from pass-specific optimization proofs.
- The committed [release-validation record](../../benchmarks/validation-2026-10-24-r6.json) records 103/103 public tests with CUDA/assembly enabled, 15/15 original workloads at each of O2 and O3, and the final emission-equivalence checks. This audit matched all 33 recorded compiler/CLI/test source hashes to the current files; it did not rerun tests, assembly, or benchmarks.

There is no dedicated public unit test for every one of the 27 entries. Some mechanisms are exercised by the original workload integration suite and their recorded compiled artifacts. Those suite results establish the documented tested scope, not exhaustive correctness for arbitrary IR or an isolated performance benefit for every pass.

## Suggested spoken-slide examples

For the narrated reuse/vectorization slide, pair **P03** (four adjacent values per lane) with **P04** (load once for the reduction, retain values for its consumer). For the reductions/split-K slide, show **P07** as lane partials → shuffle sum, **P09/P10** as a smaller communication tree, and **P06** as `S × D → (S/f) × (fD)` with the original launch retained. Keep detailed guards in this reference and the optional inventory pages rather than crowding the main slides.
