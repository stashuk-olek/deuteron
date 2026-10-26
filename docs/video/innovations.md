# Deuteron: compiler engineering contributions

These are concrete engineering contributions implemented in Deuteron, using established GPU techniques. The contribution is recognizing useful structure in scheduled IR, proving the conditions for a rewrite, and composing transformations under the existing launch and numerical contract. This is not a claim to have invented tensorization, software pipelining, or attention. [LLVM still performs scalar optimization and NVPTX generation](../../compiler/build.py#L120). The explanations below make no individual-pass speedup claims.

**1. Keep normalization inputs across the reduction boundary**

Before: a thread loads input values to compute a sum or sum of squares, participates in the reduction, then reloads the same values to normalize them. Ordinary local load elimination cannot simply cross that whole sequence without understanding ownership and effects.

After: [`forward_reduction_loads_register`](../../compiler/passes/regforward.py#L302) rewrites eligible loops into four-value loads with named temporaries that remain available to the consumer. The reduction uses those values; the consumer uses the same registers instead of repeating the global loads. Other consumer inputs and operations remain present.

The [pair matcher](../../compiler/passes/regforward.py#L161) requires compatible iteration spaces and indices, readonly input, no writes to that buffer anywhere in the tile, an intervening block barrier, and closed temporary lifetimes. It generates collision-free names and limits the number of retained vectors. The useful contribution is dataflow recovery across a collective operation, with a bounded register footprint. Reduction regrouping remains subject to the numerical contract.

**2. Fold split-K work into fewer active CTAs**

Before: multiple thread blocks compute short, disjoint K intervals for the same output tile and atomically add their partial results. Each block repeats operand staging and scheduling overhead.

After: [`fold_grid_contractions`](../../compiler/passes/gridfold.py#L23) finds a leading tile axis that changes the contraction inputs but not the output coordinates. It folds several adjacent K pieces into a deeper local contraction and reduces that logical axis. The physical grid and block dimensions stay fixed; surplus CTAs leave through the existing tile guard. Each remaining CTA performs more useful work before its atomic output update.

The [recognizer](../../compiler/passes/gridfold.py#L43) checks a closed atomic-add contraction, affine source origins, complete cooperative copies, readonly sources, and compatible dimensions. Its [factor selection](../../compiler/passes/gridfold.py#L96) also bounds shared storage and preserves a minimum amount of parallel work. The rewritten one-stage copies use ordinary vector loads/stores with the retained producer barrier. Do not describe this as universally removing atomics: this implementation retains the atomic writes while reducing the number of partial contributions.

**3. Certify and rebuild the tensor operand pipeline**

[`tensorize`](../../compiler/passes/contraction.py#L6) first recognizes dense shared-memory multiply/add contractions, including complete output ownership and supported warp tiles. BF16x3 or TF32x3 emission replaces the scalar contraction with three tensor products and FP32 accumulation.

The deeper transformation is [`stage_register_loads`](../../compiler/passes/regstage.py#L70). Before: an explicitly scheduled asynchronous copy ring stages FP32 operands, and tensor consumers load and convert operand fragments. After: vector global loads prefetch the next K tile into registers; the current tile is converted into BF16 high/residual planes in one used shared stage. Matching producer/consumer layouts feed [`ldmatrix`](../../compiler/codegen/tensor.py#L121). The [fragment loop](../../compiler/codegen/tensor.py#L133) loads the next fragments before issuing the current three MMA terms, creating a second level of lookahead.

This requires more than finding a copy instruction. The guard proves complete, nonoverlapping, aligned copies; readonly sources; bounded indices; exclusive shared-buffer ownership; and the exact original [clamped future-copy coordinates](../../compiler/passes/regstage.py#L163). A packing certificate enables the new shared interpretation only after the rewrite succeeds. The [drain](../../compiler/passes/regstage.py#L234) omits the unused final prefetch and consumer barrier, while retaining the producer barrier.

The packed BF16 path **replaces** the asynchronous ring. TF32x3 and ordinary FP32 paths retain `cp.async`; these are alternative paths. Two BF16 planes together use 32 bits per operand, so packing alone does not imply halving reserved shared memory.

**4. Recognize attention's recurrence, then reuse its work and storage**

Before: an online recurrence repeatedly updates maximum, normalization mass, and weighted-value sum as each score arrives. Every update may rescale previous state. The same query/key work can also appear across output columns.

After: [`stage_recurrences`](../../compiler/passes/recurrence.py#L47) recognizes the coupled state equations symbolically. [`RecurrenceEmitter.emit`](../../compiler/passes/recurrence.py#L609) computes scores, reduces their maximum, forms exponential weights and mass, then performs value accumulation using shared weights. Eligible [axis reordering](../../compiler/passes/recurrence.py#L17) increases lane sharing without changing the physical launch. This is recognition of a supported recurrence, not dispatch on an attention kernel's name.

Guards establish closed state use, a uniform scale, readonly operands, the expected normalized output expression, and bounded register/shared work. [Operand staging](../../compiler/passes/recurrence.py#L144) uses full coverage, estimated reuse, and a scratch budget before introducing `cp.async` copies.

There is also a specific lifetime contribution: probability-weight storage can reuse a dead staged operand allocation. [`recurrence_storage_is_warp_private`](../../compiler/passes/lifetime.py#L5) proves that overwritten reads belong to the same warp and recurrence component. The read-to-maximum-to-weight-store dependency orders reuse. A value operand still needed later, or an unproved cross-warp lifetime, keeps separate storage.

**5. Distribute PV work while preserving mutable output code**

Before: each lane walks the keys for one output column. After: [`grouped_values`](../../compiler/passes/recurrence.py#L564) forms four groups of eight lanes. Groups divide the keys; each lane loads four adjacent value columns and accumulates four partial sums. Full-warp shuffles combine the groups, then eight lanes each write four columns.

The subtle part is replaying the original output suffix. It may overwrite the denominator with its reciprocal. Replaying that code four times without restoring state would change later columns. Deuteron restores the saved denominator and current weighted sum before every replay, then restores the logical controls.

The [grouped-value guard](../../compiler/passes/recurrence.py#L390) requires one full-warp CTA, one component, 32 contiguous columns, divisible key work, FP32 buffers, distinct controls, aligned vector accesses, complete output coverage, and a closed suffix. All shuffle participants execute before the output predicate. Unsupported coordinate forms or lifetimes keep the ordinary recurrence path.

Numerically, these transformations do not promise universal IEEE equivalence. BF16x3/TF32x3 use rounded high and residual operands and omit low×low. Attention uses approximate exponentiation, and parallel reductions can change rounding. Existing tolerances remain authoritative. [FP16 global storage](../../compiler/storage.py#L5) is a separate supported subset with explicit FP32 arithmetic, exact widening and nearest-even narrowing; it is not public BF16 storage or native half tensor arithmetic.
