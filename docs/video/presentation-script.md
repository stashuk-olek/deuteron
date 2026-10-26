# Deuteron: compiler passes and innovations in three minutes

Read only the quoted narration aloud. The script has **355 spoken words**, nine narrated slides, and a five-second silent hold after 2:55. The three separate pass-reference slides are for preparation and discussion; they are outside the three-minute edit.

Record one continuous take with clear speech. The slides carry the diagrams and exact pass names; keep the narration conversational. [script.json](script.json) is the timing source. The [engineering contributions](innovations.md) and [complete pass reference](pass-reference.md) explain the implementation and guards.

| Slide | Time | Words | Main point |
|---|---|---:|---|
| 1 | 0:00–0:12 | 27 | The compiler transformations we built |
| 2 | 0:12–0:27 | 35 | Where our passes run |
| 3 | 0:27–0:44 | 36 | Reuse data before loading it again |
| 4 | 0:44–1:03 | 38 | Reshape parallel work |
| 5 | 1:03–1:26 | 43 | Turn contractions into tensor work |
| 6 | 1:26–1:49 | 44 | Rewrite the pipeline feeding tensor cores |
| 7 | 1:49–2:12 | 40 | Recognize and stage the attention recurrence |
| 8 | 2:12–2:30 | 37 | Redistribute probability–value work |
| 9 | 2:30–2:55 | 55 | Measure the whole compiler on RTX 5090 |
| 9, held | 2:55–3:00 | 0 | Silent results-slide hold |

## 0:00–0:12 · Slide 1: The compiler transformations we built

**Delivery:** Start directly. Give the three transformation families one beat each.

> Deuteron compiles scheduled Emmy Kernel IR into PTX for the RTX 5090. Our contribution is guarded compiler passes that change memory reuse, parallel work, and tensor pipelines.

**On screen:** Show the title and the three contribution families: reuse data, redistribute work, and build tensor pipelines. Keep scheduled Emmy IR and RTX 5090 visible.

## 0:12–0:27 · Slide 2: Where our passes run

**Delivery:** Follow the stages left to right. Read the early rewrite names as a sequence, not as separate claims of algorithmic invention.

> We validate IR and metadata; each rewrite checks index and ownership conditions. Early passes vectorize loops, forward reduction inputs, reorder recurrences, and fold split-K grids. Specialized lowering follows; LLVM supplies optimization and NVPTX code generation.

**On screen:** Show input validation, the actual early rewrite order, specialized lowering, and LLVM/NVPTX. The early order is vectorization, register forwarding, recurrence reordering, then grid folding. Name the project contribution: recognizers, guards, and transformed GPU schedules.

**Implemented mechanisms:** P01 · Symbolic index and effect analysis; P02 · Storage typing; P03 · Elementwise vectorization; P04 · Reduction load forwarding; P05 · Recurrence-axis reordering; P06 · Split-K grid folding.

## 0:27–0:44 · Slide 3: Reuse data before loading it again

**Delivery:** State the pattern first, then explain the change. Point to the repeated read, then show the saved values being reused.

> We look for repeated reads of the same input, or neighboring values loaded one at a time. We keep reusable values in registers and load nearby values together. That means fewer repeated trips to global memory.

**On screen:** Show scalar loop to four values per lane; then load vector, reduce, and reuse the same vector in a consumer. Highlight matching indices, readonly input, closed lifetimes, and aligned complete coverage.

**Implemented mechanisms:** P03 · Elementwise vectorization; P04 · Reduction load forwarding; P08 · Coalesced sum/square reduction.

## 0:44–1:03 · Slide 4: Reshape parallel work

**Delivery:** State the pattern first, then explain the change. Show repeated work becoming shared work, then small pieces being combined.

> We look for repeated calculations and blocks doing tiny pieces of work. We share dot products across threads, remove unnecessary reduction steps, and merge small pieces when enough parallel work remains. That reduces duplicated work and scheduling overhead.

**On screen:** Use two concrete illustrations: repeated dot to lane partials plus shuffle, and four eligible split-K partitions folded into two logical partitions with twice the K work. Label the factor-two drawing illustrative; the actual factor is guarded. Surplus physical CTAs exit; the original atomic output remains. The four partitions are per output tile within a larger grid; sufficient total parallelism remains a guard.

**Implemented mechanisms:** P06 · Split-K grid folding; P07 · Cooperative dot reduction; P09 · Small-tree elimination; P10 · Warp tree specialization.

## 1:03–1:26 · Slide 5: Turn contractions into tensor work

**Delivery:** State the pattern first, then explain the change. Follow the loop into tensor-core operations. Say that the result is approximate.

> We look for matrix multiplication written as ordinary multiply-and-add loops. We replace them with tensor-core operations, using three lower-precision products to approximate the result while adding them together in 32-bit precision. We also arrange shared memory so tensor cores can read it efficiently.

**On screen:** Show a shared-memory contraction becoming warp MMA, with high/high, high/low, and low/high products. Label low/low omitted, FP32 buffers and accumulation, and ordinary FP32 mode available. Show a separate FP16 storage to FP32 arithmetic to nearest-even FP16 store strip.

**Implemented mechanisms:** P02 · Storage typing; P11 · Contraction tensorization; P12 · Tensor shared-memory layout.

## 1:26–1:49 · Slide 6: Rewrite the pipeline feeding tensor cores

**Delivery:** State the pattern first, then explain the change. Show the next inputs arriving early, then the unused final load being removed.

> We look for tensor-core loops that repeatedly copy and convert their matrix inputs. For the supported BF16 path, we fetch upcoming inputs early and pack them before the tensor cores need them. We also skip the final prefetch when there is no more work.

**On screen:** Compare the scheduled FP32 asynchronous copy ring with the certified packed BF16 rewrite: register prefetch, producer packing, shared layout, and fragment lookahead. Keep the exact copy-phase guard visible. Show final unused prefetch/consumer barrier omitted while producer synchronization remains. TF32x3 and FP32 retain the eligible cp.async path.

**Implemented mechanisms:** P12 · Tensor shared-memory layout; P13 · Packed register staging; P14 · Tensor fragment pipelining; P15 · Pipeline drain.

## 1:49–2:12 · Slide 7: Recognize and stage the attention recurrence

**Delivery:** State the pattern first, then explain the change. Show one score calculation feeding several outputs, then temporary storage being reused.

> We look for attention loops that repeat the score calculation across output columns. We calculate scores once per group and reuse their weights. We also load neighboring keys together and reuse temporary storage once its values are no longer needed.

**On screen:** Show repeated score and online max/mass/value work becoming score reuse, row maximum, shared unnormalized weights, and separate mass/PV accumulation. Show two/four adjacent key rounds and read-to-maximum-to-store lifetime ordering in the same warp and component. Each unnormalized weight feeds register mass accumulation directly and shared storage for PV; mass does not reread shared weights.

**Implemented mechanisms:** P16 · Staged online recurrence; P17 · Blocked recurrence loads and address hoisting; P19 · Recurrence shared-storage reuse.

## 2:12–2:30 · Slide 8: Redistribute probability–value work

**Delivery:** State the pattern first, then explain the change. Show the keys split among groups, then their partial results being combined.

> We look for attention loops where each thread scans every key for one output column. We split keys among thread groups, load four adjacent columns together, and combine their partial results. Each column keeps its original normalization.

**On screen:** Draw four eight-lane groups, each processing a key shard and four adjacent columns per lane. Combine groups with butterfly offsets 16 and 8; only the first eight lanes write. Show restoring the saved denominator before each closed per-column suffix replay.

**Implemented mechanisms:** P18 · Warp-distributed vector PV reduction.

## 2:30–2:55 · Slide 9: Measure the whole compiler on RTX 5090

**Delivery:** Keep warm, tested baseline, whole-compiler, and limited search. Do not attribute the aggregate numbers to a single pass.

> On fifteen open workloads, geometric mean warm latency is three point seven five microseconds. Speedups are one point six six times over generated CUDA with NVCC, and one point one five times over the tested Triton baseline. All two hundred seventy calls passed the original checks. These are whole-compiler results with a limited Triton search.

**On screen:** Show the recorded warm latency bars: Deuteron 3.752 microseconds, NVCC 6.216, tested Triton 4.332. Show 1.66x and 1.15x, 15 open workloads, 270 balanced calls, 20 warmups/200 iterations, original output checks, and compile/assembly/transport excluded. Hold this slide unchanged for the last five seconds.

## What to call innovative

Present the contribution as the compiler recognizers, legality checks, ownership rewrites, and composition of GPU transformations. The exact copy-phase proof, packed producer/consumer certificate, guarded attention storage reuse, and mutable-suffix replay are concrete engineering details. These use established GPU algorithms and instructions. LLVM supplies its optimization and NVPTX backend.

The optional [three-page pass reference](pass-reference.pdf) covers all 27 documented entries. That count includes analyses, policies, and lowering mechanisms; it does not mean 27 independent pass modules. The [source-linked explanations](pass-reference.md) give the before/after behavior and guards.

## Pronunciation

| Written | Say |
|---|---|
| IR / PTX | “I–R” / “P–T–X” |
| LLVM / NVPTX | “L–L–V–M” / “N–V–P–T–X” |
| BF16x3 | “bee-eff sixteen times three” |
| TF32x3 | “tee-eff thirty-two times three” |
| FP16 / FP32 | “eff-pee sixteen” / “eff-pee thirty-two” |
| MMA / CTA / PV | Spell the letters |
| Split-K | “split kay” |
| cp.async | “see-pee ay-sync” |
| RTX 5090 | “R–T–X fifty-ninety” |

## Optional cuts after rehearsal

- Slide 3: omit the final sentence about fewer trips to memory; the diagram shows the benefit.
- Slide 5: omit the final sentence about arranging shared memory; the diagram still shows the layout.
- Slide 6: omit the final sentence about skipping the last prefetch; the diagram still shows this change.

Keep the tensor approximation, LLVM attribution, and benchmark qualifiers. The slide diagrams and evidence notes retain the detailed rewrite conditions. If the take runs long, use a shorter sentence or pickup rather than speeding up technical names.

## Evidence and boundaries

- The [round-6 comparison](../../benchmarks/rtx5090-2026-10-24-r6.json) has 270 balanced warm calls over 15 open workloads. Geometric means are 3.752 µs Deuteron, 6.216 µs generated CUDA/NVCC, and 4.332 µs tested Triton. These are whole-compiler results, with a limited Triton search and original checks. Compilation, assembly, and transport are excluded.
- The split-K illustration shows four partitions per output tile within a larger grid. Factor selection preserves enough total parallel work, bounded K depth, and scratch capacity. The physical grid and atomic output updates remain.
- The packed BF16 rewrite replaces an FP32 copy ring. Eligible TF32x3 and FP32 paths retain asynchronous copies. “One used shared stage” is not a claim that reserved shared memory halves.
- BF16/TF32 are internal approximate tensor formats, with FP32 buffers and accumulation. FP16 storage is a separate pointwise subset with explicit FP32 arithmetic. Reductions can change association.
- Recurrence weights are unnormalized. Each weight directly updates register mass and is also stored for PV. Shared-storage aliasing needs same-warp/component dependence; grouped PV restores the denominator for each suffix replay.
- The [deployment contract](../production.md) describes the supported scheduled-IR subset. TMA, WGMMA, TMEM, public BF16 buffers, and native half arithmetic are outside the implementation.

The previous standalone PTX-demo and reliability/test-count slides are removed from this presentation. The optional captured demo remains in the downloadable source kit as supporting material, outside the narrated deck.

At **2:55**, stop speaking and hold slide 9 unchanged until **3:00**. The [recording and editing guide](recording-and-editing.md) describes the planned postproduction.
