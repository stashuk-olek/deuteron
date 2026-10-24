# Deuteron

Deuteron compiles **scheduled Emmy Kernel IR to PTX**, using guarded GPU transformations, typed LLVM IR, and LLVM's NVPTX backend. It is validated on the 15 open Emmy workloads on H100 and RTX 5090. Compilation requires no GPU, CUDA toolkit, PyTorch, or Triton.

Features to look for in this submission:

| Feature | Implementation and example |
|---|---|
| Tensor cores | Compensated **BF16x3 and TF32x3** products with FP32 accumulation; packed BF16 operands, `ldmatrix`, and `mma.sync`. See [tiled matrix multiplication](examples/tensor_matmul/README.md). |
| Asynchronous copies | Global-to-shared `cp.async` transfers of 4, 8, or 16 bytes, commit/wait groups, and CTA/warp barriers. The matrix example's `f32` and `tf32x3` modes retain this path. |
| FP16 storage | Scalar/vector `f16` loads and stores, exact widening, and round-to-nearest-even narrowing, with explicit FP32 arithmetic. See [half-storage examples](examples/f16_storage.py). |
| Reductions and attention | Cooperative dot products, coalesced reductions, register forwarding, staged softmax recurrences, warp maximum reduction, shared-storage reuse, and vectorized probability–value reduction. |
| Inspectable compilation | PTX and LLVM IR before/after optimization, source-based caching, standalone examples, and CPU/optional CUDA tests. |
| Deployment | Installable `deuteron` command, restricted IR parsing, validated options/launches, bounded memory cache, atomic persistent cache, runtime error checks, and CPU CI. See the [deployment contract](docs/production.md). |

[SUBMISSION.md](SUBMISSION.md) gives a short feature checklist. The [performance comparison](#rtx-5090-performance) includes NVCC and Triton baselines, source, and raw measurements.

## Build and run

Requires Python 3.12 or newer; Python 3.12/3.13 are covered by the CPU CI configuration:

```sh
python -m pip install .
deuteron --version
deuteron examples/vector_add/vector_add.ir.json --arch sm_120 > /tmp/vector_add_5090.ptx
python compile.py examples/vector_add/vector_add.ir.json > /tmp/vector_add.ptx
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json --arch sm_120 > /tmp/matmul.ptx
```

The adjacent `meta.json` defines argument order, buffer types/shapes, launch geometry, and shared-memory size. The compiler reads the IR and metadata; expected outputs and reference implementations are used only by test and benchmark tools.

Keep both LLVM stages for inspection:

```sh
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json \
  --arch sm_120 --dump-llvm /tmp/matmul.ll \
  --dump-unoptimized /tmp/matmul.unoptimized.ll > /tmp/matmul.ptx
```

The CLI defaults to SM89/O3; the Python API defaults to SM89/O2. Accepted targets are `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_110`, and `sm_120`. Target support in the installed assembler/driver is a separate requirement; native SM120 assembly was checked with CUDA 12.8. Emission uses PTX 8.0 below SM100, PTX 8.7 for SM100/SM120, and PTX 9.0 for SM110. SM110 emission is tested without a GPU; its assembly/execution requires a compatible newer toolchain and hardware.

```python
from pathlib import Path
from compiler import Options, compile_kernel

result = compile_kernel(
    Path("examples/tensor_matmul/matmul_tiled_128.ir.json"),
    arch="sm_120", opt_level=3,
    options=Options(contraction_precision="bf16x3"),
)
print(result.ptx)
```

`compile_many()` compiles independent jobs in worker processes; calls in one process serialize LLVM work. The memory cache is bounded, and `clear_cache()` clears it. Set `DEUTERON_CACHE` to a directory for persistence. Versioned records use integrity checks and atomic writes; corrupt or inaccessible caches fall back to compilation. Cache keys include compiler source, dependency versions, IR and metadata bytes, target, optimization level, and every compiler option. Restart the process after editing compiler source.

## Examples and tests

The examples are self-contained and use independent inputs and references:

| Example | What it demonstrates |
|---|---|
| [Vector add](examples/vector_add/vector_add.ir.json) | 1,030 FP32 elements, two elements per thread, and a partially populated final block. Exact NumPy reference. |
| [Tiled matmul](examples/tensor_matmul/README.md) | 128 × 128 matrices, BF16x3/TF32x3 tensor cores, `ldmatrix`, and a three-stage asynchronous-copy ring. Float64 reference of FP32 inputs. |
| [FP16 storage generator](examples/f16_storage.py) | Copy, widen, narrow, affine arithmetic, and an FP32-intermediate cancellation example. Generates IR, metadata, references, and PTX. |
| [Grouped attention](examples/attention.md) | Independent attention shapes and in-place normalization suffixes for the warp-distributed probability–value pass. |

```sh
python examples/f16_storage.py --output /tmp/f16-examples
python examples/attention.py --output /tmp/attention-example
python -m unittest discover -s tests -v
PTXAS=/usr/local/cuda/bin/ptxas python -m unittest discover -s tests -v
```

CPU tests check restricted parsing, options and ABI validation, concurrent/corrupt cache behavior, CLI file handling, pass legality, runtime failures, and semantic counterexamples. CUDA tests are opt-in; choose an available GPU explicitly:

```sh
DEUTERON_TEST_DEVICE=1 python -m unittest discover -s tests -v
python examples/f16_storage.py --output /tmp/f16-examples --device 1
python benchmark.py vector_add tensor_matmul --problems-dir examples --devices 1
```

The half tests cover all 65,536 bit patterns for raw copies and widening, and 190,468 FP32 inputs at and around half rounding boundaries. They also check unaligned half/float vector accesses and explicit FP32 intermediates. Tensor tests check emitted instructions, assembly, all three precision modes, and a wrong-prefetch-phase counterexample. Attention tests check ownership, guard rejection, and suffix replay, including an assignment that overwrites the denominator.

| Interface tests | What they check |
|---|---|
| [Validation](tests/test_validation.py) | Constructor escapes never execute; malformed IR, options, launch dimensions, duplicate ABI entries, and writes to readonly buffers fail explicitly. |
| [Cache](tests/test_cache.py) | Bounded memory use, corruption recovery, atomic disk records, threaded compilation, spawned workers, and validated batch jobs. |
| [Runtime](tests/test_runtime.py) | Context ownership, input/response validation, bounded busy retries, stale-cubin recovery, scalar-buffer handling, and preservation of earlier artifacts. |
| [CLI](tests/test_cli.py) | Command options/defaults, diagnostics, dump-path collisions, atomic output handling, broken pipes, and package dependency consistency. |

[Release validation](benchmarks/validation-2026-10-24-r6.json) passes **103/103 public tests on Python 3.13**, with CUDA and assembly enabled and no skips. Python 3.12 CPU discovery runs 103 tests with 11 expected CUDA/assembly skips and no failures. The original workloads pass 15/15 at O3 and 15/15 at O2; local suites use GPUs 1–7 with clean occupancy checks, excluding GPU 0. All [11 native SM120 RTX 5090 feature examples](benchmarks/features-rtx5090-2026-10-24-r6.json) also pass. The final source reproduces all 45 audited PTX files and 90 LLVM stages byte for byte, and all 75 assembled cubins match. [Earlier validation](benchmarks/validation-2026-10-24.json) remains available. The fresh comparison below has completed all 270 calls with correct outputs.

## How compilation works

The input already describes a GPU schedule: tile/loop axes, scalar operations, explicit shared buffers, copies, collectives, and barriers. Deuteron validates that schedule and recognizes structures that can use a better GPU lowering while preserving its launch and output contract.

```mermaid
flowchart LR
    A["Scheduled Emmy IR + meta.json"] --> B["Restricted parser + validation + structural passes"]
    B --> C["Typed LLVM IR + NVVM / PTX instructions"]
    C --> D["SROA + LLVM optimization + cleanup"]
    D --> E["NVPTX emission + PTX rewrites"]
    E --> F["CUDA assembler / driver → GPU"]
```

1. [build.py](compiler/build.py) reads one `KernelOp` through a restricted constructor parser, matches and validates its launch metadata/options/storage types, and applies vectorization, load forwarding, recurrence reordering, and split-K folding. Serialized IR is never evaluated as arbitrary Python.
2. [lower.py](compiler/codegen/lower.py) recognizes specialized normalizers, packed matrix staging, and reductions. It builds typed global/shared pointers and lowers ordinary operations or certified tensor/recurrence plans. NVVM intrinsics and small inline PTX operations express shuffles, barriers, copies, and tensor instructions.
3. LLVM verifies the IR. SROA promotes temporary scalar slots; the selected LLVM pipeline performs scalar and loop optimization. Final cleanup runs loop strength reduction for striped reductions, NewGVN, instruction combining, and CFG simplification.
4. NVPTX emits assembly. [ptx.py](compiler/ptx.py) performs the final atomic, block-contract, and optional division rewrites. Assembly and GPU execution happen separately.

Transformations inspect operations, expressions, shapes, ownership, and effects. They do not dispatch on benchmark names or input values. A failed optimization proof keeps the existing lowering; unsupported IR or storage semantics produce an explicit error.

## Compiler passes

This inventory includes structural passes and optimization mechanisms implemented inside lowering. LLVM's own pipeline is listed separately below.

| Pass / source | Transformation and main guard |
|---|---|
| [Symbolic index and effect analysis](compiler/analysis.py) | Canonicalizes supported affine expressions, checks uses/equality/strides, and provides inputs to pass-specific bounds and ownership proofs. Unsupported expressions stop the relevant rewrite. |
| [Storage typing](compiler/storage.py) | Validates half/float types before rewrites and lowering: explicit FP32 arithmetic, control-only indices, and ordinary output stores. |
| [Elementwise vectorization](compiler/passes/vectorize.py) | Converts contiguous thread-strided scalar loops to four elements per lane. Requires divisible extents, safe overlap, closed temporaries, and ordinary stores. |
| [Reduction load forwarding](compiler/passes/regforward.py) | Keeps input vectors in registers across a reduction and its consumer, removing repeated global loads. Proves matching indices, readonly inputs, and closed lifetimes. |
| [Recurrence-axis reordering](compiler/passes/recurrence.py) | Reorders logical tile axes to increase provable lane sharing while retaining the physical launch and output mapping. |
| [Split-K grid folding](compiler/passes/gridfold.py) | Folds an output-independent atomic contraction axis into deeper per-CTA K work. Guards affine coverage, shared memory, and remaining parallelism; surplus physical CTAs exit through the tile guard. |
| [Cooperative dot reduction](compiler/transforms.py) | Distributes a repeated dot product across a power-of-two lane group and combines partial sums with shuffles. Requires full participating warps and closed scalar/index scope. |
| [Coalesced sum/square reduction](compiler/transforms.py) | Maps thread-strided reductions to contiguous vector loads. Long eligible loops use independent FP32 accumulators and alternating load batches; explicit unroll settings bypass that automatic pipeline. |
| [Small-tree elimination](compiler/passes/tailtree.py) | Replaces closed trees of 2–8 warp partials with direct shared loads and balanced scalar addition; eligible multiple-state trees use a subgroup shuffle. Requires the producer barrier and exclusive scratch lifetime. |
| [Warp tree specialization](compiler/codegen/lower.py) | Lets warp 0 combine up to 32 warp partials and broadcast through shared memory with one CTA barrier. Other trees retain generic shared-memory reduction. |
| [Contraction tensorization](compiler/passes/contraction.py) | Recognizes dense shared-memory multiply/add contractions and emits BF16x3 or TF32x3 MMA. Proves complete affine output tiles, full warps, supported dimensions, and scalar ownership. |
| [Tensor shared-memory layout](compiler/codegen/tensor.py) | Applies matching producer/consumer bank remapping and packed layouts; uses `ldmatrix` for eligible BF16 fragments. |
| [Packed register staging](compiler/passes/regstage.py) | Replaces a certified asynchronous matrix-copy ring with vector global prefetches, BF16 high/residual packing, and one packed shared stage. Checks original clamped future coordinates, alignment, bounds, and ownership. |
| [Tensor fragment pipelining](compiler/codegen/tensor.py) | Carries current MMA fragments in loop phis and loads the next fragment before consuming the current one. Keeps the fragment loop compact; the outer packed K loop defaults to unroll two. |
| [Pipeline drain](compiler/passes/regstage.py) | Omits the final unused register prefetch and consumer barrier when no next K tile exists. Producer stores and their barrier remain; the staging proof excludes later shared users. |
| [Staged online recurrence](compiler/passes/recurrence.py) | Recognizes max/mass/weighted-sum recurrences, computes scores once, and separates normalization from value accumulation using shared probabilities. May stage reusable readonly operands with `cp.async` under traffic/scratch-budget guards. |
| [Blocked recurrence loads and address hoisting](compiler/passes/recurrence.py) | Groups contiguous key rounds for vector loads and hoists affine value-gather bases. Requires full coverage, compatible staging, and bounded positive strides. |
| [Warp-distributed vector PV reduction](compiler/passes/recurrence.py) | For certified single-warp, 32-column recurrences, four groups distribute keys while each lane accumulates four adjacent columns. Shuffles combine groups. Proves alignment, unique output ownership, and closed per-column suffix replay; restores the denominator each time. |
| [Recurrence shared-storage reuse](compiler/passes/lifetime.py) | Reuses dead staged-operand storage for probabilities when every overwritten read belongs to the same warp/component. Read-to-maximum-to-store dependence orders reuse; otherwise scratch remains separate. |
| [Hybrid row normalizer](compiler/passes/hybrid_normalizer.py) | Each warp obtains the row maximum, each thread exponentiates its owned values, and only scalar mass crosses warps. Requires a closed max/mass tree, bounded work, and approximate math enabled. |
| [Ordered hardware warp maximum](compiler/codegen/lower.py) | Maps proven non-NaN FP32 values to ordered integer keys for `redux.sync.max.s32`, then reverses the mapping. Used for full warps in certified normalizers and multi-warp recurrences; single-warp recurrences retain shuffles. |
| [CTA/thread address factoring](compiler/codegen/lower.py) | Separates leading CTA axes from trailing thread axes when their product matches the block width. Avoids repeated full-ticket division; guards bounds, rasterization, and partial-grid ownership. |
| [Loop cost policy](compiler/transforms.py) | Supplies structure-sensitive unroll/vectorization hints. Explicit options remain authoritative; this is a heuristic policy, not runtime autotuning. |
| [Reciprocal-square-root folding](compiler/transforms.py) | Folds adjacent sqrt→reciprocal dataflow to approximate reciprocal square root when enabled. |
| [Dead atomic-result elimination](compiler/ptx.py) | Changes unused-result atomic add to PTX reduction while retaining recognized predicate/order/scope qualifiers. |
| [Fixed block contract](compiler/ptx.py) | Changes `.maxntid` to `.reqntid` for the launch's fixed thread dimensions. |
| [Approximate division cleanup](compiler/ptx.py) | Selects approximate division lowering only when the corresponding math option is enabled. |

The explicit LLVM stages are **SROA**, the LLVM O0–O3 pipeline, optional loop/SLP vectorization, conditional **loop strength reduction**, **NewGVN**, **InstCombine**, and **SimplifyCFG**, followed by NVPTX instruction selection/code generation. LLVM supplies its own internal passes, separately from Deuteron's structural transformations. The `warp_willreturn` option also marks supported warp intrinsics as returning, helping LLVM reason about their control flow.

## Types, precision, and supported scope

The full scheduled backend uses FP32 global/shared values. Buffer arguments use distinct, 16-byte-aligned allocations and must follow the metadata's argument order and launch geometry. Full-FP32 vector offsets must be naturally aligned to the vector width. FP16 global storage supports `Tile`, `Loop`, `StridedLoop`, scalar/vector loads, explicit `Assign(dtype=F32)`, and ordinary stores. Mixed half/float accesses use conservative scalar alignment, including vectors with unaligned origins within those buffers. Raw half copies preserve bits; widening is exact; narrowing rounds to nearest with ties to even. Converted NaN payloads are unspecified. The [deployment contract](docs/production.md) spells out caller obligations and validation limits.

BF16 and TF32 are **internal tensor operand formats**. Each FP32 operand is split into a rounded high part and residual; three products—low×high, high×low, high×high—accumulate into FP32. The low×low term is omitted. These modes approximate FP32 multiplication and do not change benchmark tolerances. Select `--contraction-precision f32` for ordinary FP32 products, or `tf32x3` for the alternative tensor mode.

`--no-approximate-math` disables optional reciprocal/sqrt/division approximations. Exponentiation still uses approximate base-2 GPU instructions, so this switch is not a universal IEEE mode. Existing arithmetic options also apply to explicit FP32 arithmetic in half-storage kernels. Approximate modes are validated for supplied inputs; they do not promise IEEE equivalence for arbitrary infinities, overflow, or subnormals.

Public BF16 buffers, native half arithmetic/atomics, half shared-memory/tensor paths, scalar kernel parameters, nonlinear launches, and serial/auxiliary thread axes are outside the supported subset. TMA, WGMMA, and TMEM lowerings are not implemented. SM120 emission uses the supported MMA/copy instruction families; it does not imply use of Blackwell-specific tensor instructions.

## RTX 5090 performance

On these 15 workloads, Deuteron achieves **1.66× the speed of generated CUDA/NVCC** and **1.15× the speed of this Triton baseline**, using the equal-weight geometric mean of warm latency. Measurements below are in **microseconds; lower is better**, remeasured on the RTX 5090 on 2026-10-24 in round 6. Deuteron targets native SM120/O3.

| Workload | Deuteron | NVCC 12.8.93 | Triton 3.6 |
|---|---:|---:|---:|
| p00 Add, 128 | 2.976 | 2.976 | 3.104 |
| p01 Add, 1024 | 2.976 | 2.976 | 3.104 |
| p02 Affine, 4096 | 3.040 | 3.040 | 3.200 |
| p03 ReLU × multiply, 2048 | 2.976 | 2.992 | 3.104 |
| r01 Sum, 1024 | 2.960 | 2.976 | 3.048 |
| r02 Sum, 65536 | 6.816 | 10.728 | 8.880 |
| r03 Mean, 32768 | 4.784 | 6.848 | 4.832 |
| r04 Row sum, 64 × 512 | 2.960 | 3.040 | 3.096 |
| m01 Matmul, 512³ | 8.896 | 15.040 | 12.992 |
| m02 Matmul, 128 × 256 × 64 | 3.144 | 5.088 | 6.848 |
| f01 RMSNorm, 64 × 1024 | 3.152 | 6.848 | 3.280 |
| f02 Softmax, 64 × 256 | 2.976 | 2.976 | 3.072 |
| f03 L2Norm, 64 × 512 | 2.976 | 4.744 | 3.104 |
| a01 Attention, 64 rows / 64 depth | 4.800 | 64.256 | 4.832 |
| a02 Attention, 128 rows / 32 depth | 4.800 | 31.424 | 6.816 |
| **Geometric mean** | **3.752** | **6.216** | **4.332** |

Where we stand: the largest gains over direct NVCC references are attention (13.39× and 6.55×). Against this Triton baseline, rectangular matmul is 2.18× faster, 512³ matmul 1.46×, and the 128-row attention case 1.42×. Many pointwise and short-row kernels sit near the roughly 3 µs launch floor; differences of a few hundredths of a microsecond should be treated as ties. This establishes competitive performance on the open set, not peak throughput or a ranking against fully tuned libraries.

All compilers use the same original inputs, constants, expected outputs, and tolerances. Deuteron and NVCC preserve supplied launches; Triton uses independently written kernels and its own launch geometry. NVCC compiles Emmy's generated CUDA references at O3 with default math. Triton 3.6 uses FP32 buffers, BF16x3 dot products, and default FP fusion. Deuteron uses its default guarded tensor/approximate-math policies. Attention therefore uses different dot-product policies across implementations. Only the original atomic rectangular matmul requires output resets.

Each value is the median of six held-out hosted-runner measurements at 20 warmups and 200 iterations. Every problem sees all six compiler execution orders; configuration-screen samples are excluded and no timing sample is deleted. Every call checks the complete original output. Compilation, transport, and assembly time are excluded. These are **warm kernel timings**, not the competition's cold-launch score or an end-to-end model benchmark. Differences near the runner's timing granularity are not significance claims.

The initial Triton screen used three predeclared configurations for each attention/matmul and one for other kernels. Round 6 keeps those selected configurations and exact PTX frozen, then measures a new balanced schedule. This is a reproducible baseline, not an exhaustive Triton autotune or a comparison against cuBLAS/cuDNN. See [benchmark reproduction and limitations](benchmarks/README.md), [new raw results](benchmarks/rtx5090-2026-10-24-r6.json), and the preserved [previous measurements](benchmarks/rtx5090-2026-10-24.json). The hardening changes retain identical generated kernels; small differences between measurement rounds are not new optimization gains.

The attention pass added in round 5 reduces the SM120 cubin assembled from default SM89/O3 CLI PTX from 15,144 to 14,280 bytes and registers from 167 to 80. Its [historical fixed 100-call RTX 5090 confirmation](benchmarks/attention-confirmation-2026-10-24.json) found warm latency unchanged at 4.800 µs. The cold effect estimate was +0.0067 µs with a 95% randomization interval of [−0.2027, +0.2920] µs: a measured resource reduction, with no confirmed latency gain.

The separate [round-6 single-launch profile](benchmarks/cold-rtx5090-2026-10-24-r6.json) retains 90 correct calls at zero warmups and one iteration, using default CLI SM89/O3 PTX assembled for SM120. Its geometric mean of per-workload medians is 19.648 µs. This profile uses a different launch regime from the warm table and makes no comparative speedup claim.

## Benchmark your checkout

```sh
python benchmark.py --problems-dir /path/to/emmy-hackathon-2026/problems --devices 1,2
python benchmark.py --problems-dir /path/to/emmy-hackathon-2026/problems --remote --arch sm_120
```

Local execution needs an NVIDIA GPU and CUDA driver; remote execution needs `curl`. Add problem names to select a subset. Results include timings, source hashes, PTX, both LLVM stages, and actual GPU outputs in `benchmark-results/`. Each run requires a new artifact directory to preserve earlier results. Keep local H100 and hosted RTX 5090 measurements separate, and choose a local target matching your GPU.

The runtime validates inputs before device/network work, checks remote outputs and timings, bounds busy retries, and recompiles once for an explicitly expired cubin. A local runner must be used and closed by its creating process and thread. See the [runtime and deployment contract](docs/production.md#runtime-and-measurements) for ownership, reset, and error-handling requirements.

The compiler reuses LLVM for optimization and code generation. Eligibility under the hackathon's broad prohibition on existing compilers remains unconfirmed. The validated benchmark scope is the 15 open problems; hidden-set support is not claimed.
