# Deuteron: three-minute presentation script

Read only the quoted narration aloud. The full take is **359 spoken words** across 175 seconds, followed by a five-second silent hold. That is about 123 words per minute; acronym pronunciation adds a little time. The matching narration and slide cues are in [script.json](script.json).

Record a clear, conversational take. Leave small gaps between sentences so transitions can be edited later. Music and transitions are not needed in the raw recording.

| Slide | Time | Words | Main point |
|---|---|---:|---|
| 1 | 0:00–0:20 | 40 | Scheduled IR to RTX 5090 |
| 2 | 0:20–0:50 | 61 | Structure, guards, and LLVM |
| 3 | 0:50–1:25 | 70 | Tensor cores, copies, and types |
| 4 | 1:25–1:50 | 51 | Compile it, inspect it |
| 5 | 1:50–2:30 | 85 | Measured on the open set |
| 6 | 2:30–2:55 | 52 | Reliability and an explicit scope |
| 6, held | 2:55–3:00 | 0 | Silent end hold |

## 0:00–0:20 · Slide 1: Scheduled IR to RTX 5090

**Delivery:** Start directly, with a small pause after “good schedule.” Look toward the camera for the hook, then let the title carry the hardware name.

> A fast GPU still needs a good schedule. We built Deuteron to compile scheduled Emmy Kernel IR into PTX for the RTX 5090. We reduce repeated work and memory traffic while preserving each workload's original launch, inputs, and numerical tolerances.

**On screen:** Show Deuteron, Scheduled Emmy Kernel IR → PTX, and RTX 5090. Reveal a short line: same launch, inputs, and tolerances. Keep the opening uncluttered.

## 0:20–0:50 · Slide 2: Structure, guards, and LLVM

**Delivery:** Follow the arrows as they appear. Stress “checks” and give the LLVM credit its own breath. The three examples are enough; do not list the entire pass inventory.

> First, we validate the kernel and launch metadata. Custom passes then recognize reusable loads, cooperative reductions, and matrix pipelines. Each rewrite checks shapes, ownership, and dependencies; if its proof fails, supported code keeps the generic path. We lower the result into typed LLVM IR. LLVM supplies optimization and the NVPTX backend; our contribution is the GPU analysis, guarded transformations, and lowering.

**On screen:** Reveal the pipeline left to right: IR + launch metadata → validation and custom GPU passes → typed LLVM IR → LLVM optimization and NVPTX → PTX. Highlight reuse, reductions, and matrix staging. Keep the LLVM/NVPTX credit visible.

## 0:50–1:25 · Slide 3: Tensor cores, copies, and types

**Delivery:** Use the three columns as beats. Say the tensor names slowly, then pause before the copy and storage features. Keep “internal” and “approximations” audible.

> For matrix multiplication, we use compensated BF16x3 or TF32x3 tensor products. Buffers and accumulation stay FP32; lower precision is internal, and these modes remain approximations. Ordinary FP32 multiplication is available too. Asynchronous global-to-shared copies, with explicit waits and barriers, let computation overlap data movement. A separate FP16 storage path provides exact widening and nearest-even narrowing, with explicit FP32 arithmetic. The focus is reducing memory traffic and feeding the hardware effectively.

**On screen:** Use three columns: BF16x3 / TF32x3 tensor products, cp.async + waits/barriers, and FP16 storage → FP32 arithmetic. Label tensor buffers and accumulation FP32, internal approximate operands, and the ordinary FP32 option. Optional small copy label: 4 / 8 / 16 bytes.

## 1:25–1:50 · Slide 4: Compile it, inspect it

**Delivery:** Switch to the prepared terminal capture at 1:25. Let the two highlighted instruction lines remain readable while you speak. Use the real captured output; there is no need to type during narration.

> Let's compile the standalone matrix example in TF32x3 mode for the 5090. Compilation runs on the CPU, using only the scheduled IR and its adjacent launch metadata. The generated PTX contains mma.sync tensor instructions and cp.async copies. These are actual compiler outputs. This command only compiles; GPU correctness is tested separately.

**On screen:** Show the real terminal capture from docs/video/demo/capture/transcript.txt. Recording command: python docs/video/demo/run.py --output /tmp/deuteron-video-demo. Highlight the emitted mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 and cp.async.cg.shared.global instructions. Keep a Compile-only · SM120/O3 · TF32x3 label visible; do not imply live GPU execution or numerical checking.

## 1:50–2:30 · Slide 5: Measured on the open set

**Delivery:** Slow slightly for the two ratios. Keep “warm,” “tested Triton baseline,” and “fifteen open workloads” in the take. Point to the bars once, then leave the chart still.

> Across fifteen open workloads, Deuteron's geometric mean warm latency is three point seven five microseconds. That's a one point six six times speedup over generated CUDA compiled with NVCC, and one point one five times over the tested Triton baseline. We used two hundred seventy calls with balanced execution order, checking every output against the original references and tolerances. Raw samples and artifacts are included. Compilation, assembly, and network time are excluded. These are warm kernel timings on this set, with a limited Triton search.

**On screen:** Show warm geometric-mean latency bars: Deuteron 3.752 µs, NVCC 6.216 µs, tested Triton 4.332 µs. Headline Deuteron speedups: 1.66× / 1.15×. Footer: 15 open workloads · 270 balanced warm calls · all original output checks pass. Small method detail: 20 warmups / 200 iterations per call. Link the round-6 raw results; do not mix in the separate cold profile.

## 2:30–2:55 · Slide 6: Reliability and an explicit scope

**Delivery:** Separate the H100 and RTX test counts with a short pause. Finish the final sentence by 2:55; hold the closing slide silently until 3:00.

> We also ship an installable command, restricted parsing, validated interfaces, and corruption-tolerant caches. All one hundred and three tests passed with GPU checks enabled on H100; eleven feature examples passed on the 5090. Scope stays explicit: supported scheduled IR and documented numerical modes. Deuteron makes GPU compiler decisions visible, reproducible, and measurable.

**On screen:** Show restricted parsing, validated interfaces, bounded memory cache, and atomic checked disk records. Separate validation labels: 103 tests with GPU checks enabled on H100; 11 native RTX 5090 examples. Close with Supported scheduled IR · Reproducible evidence and a readable repository label. Keep this slide unchanged for the five-second silent end hold.

## Pronunciation

| Written | Say |
|---|---|
| PTX | “P–T–X” |
| LLVM / NVPTX | “L–L–V–M” / “N–V–P–T–X” |
| BF16x3 | “bee-eff sixteen times three” |
| TF32x3 | “tee-eff thirty-two times three” |
| FP16 / FP32 | “eff-pee sixteen” / “eff-pee thirty-two” |
| cp.async | “see-pee ay-sync” |
| mma.sync | “em-em-ay sync” |
| RTX 5090 / H100 | “R–T–X fifty-ninety” / “H one hundred” |

Say “one point six six times speedup” and “one point one five times” calmly. The slide shows the exact values, so do not add more decimal places aloud.

## Optional cuts if rehearsal runs long

Cut whole sentences and keep the original segment boundaries. Use the saved time for pauses; do not accelerate the results or LLVM credit.

- Slide 3: omit “Ordinary FP32 multiplication is available too.” The option can stay on the slide.
- Slide 3: omit “The focus is reducing memory traffic and feeding the hardware effectively.” This saves about five seconds.
- Slide 4: omit “These are actual compiler outputs.” Keep the compile-only explanation and both real instruction lines.

Keep the benchmark qualifiers, the internal/approximate tensor-format explanation, and the distinction between H100 tests and RTX examples. No cut should turn a supported-subset claim into a general one.

## Facts to keep consistent during recording

- The demo is CPU compilation of the standalone matrix example in TF32x3 mode. The [captured transcript](demo/capture/transcript.txt) contains actual emitted instructions; it is not a GPU correctness or timing run.
- The [round-6 comparison](../../benchmarks/rtx5090-2026-10-24-r6.json) contains 270 balanced warm calls on 15 open workloads. Geometric-mean latencies are 3.752 µs, 6.216 µs, and 4.332 µs. This is a comparison with generated CUDA/NVCC and a tested, limited-search Triton baseline; the latest interface-hardening round did not introduce a kernel speedup.
- The [release test record](../../benchmarks/validation-2026-10-24-r6.json) covers 103 tests with CUDA checks enabled on H100. The [eleven native RTX 5090 feature runs](../../benchmarks/features-rtx5090-2026-10-24-r6.json) are separate evidence. Do not describe all 103 as RTX GPU executions.
- BF16 and TF32 are internal tensor operands with FP32 buffers and accumulation. FP16 support is storage and conversion with explicit FP32 arithmetic. TMA, WGMMA, TMEM, public BF16 buffers, and native half arithmetic are not implemented.
- Deuteron uses LLVM optimization and NVPTX code generation. Its scope is supported scheduled Emmy Kernel IR, not arbitrary graphs or a proven hidden workload set. The [deployment contract](../production.md) explains validation limits and numerical modes; the [README](../../README.md) preserves the full pass inventory and evidence.

At **2:55**, stop speaking. Keep slide 6 visible and unchanged until **3:00**. The final five seconds are intentionally silent.

