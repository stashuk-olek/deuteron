# Deuteron submission

Deuteron lowers the supported subset of scheduled Emmy Kernel IR through custom GPU passes and LLVM NVPTX to executable PTX. Its focus is RTX 5090 performance while retaining the original launch, inputs, and numerical tolerances. See the [architecture and complete pass inventory](README.md#how-compilation-works) and [deployment contract](docs/production.md).

## Features to inspect

| Feature | Where to see it |
|---|---|
| BF16x3 tensor cores | [128³ matrix example](examples/tensor_matmul/README.md): `cvt.rn.bf16x2`, `ldmatrix`, and `mma.sync.aligned.m16n8k16`, with FP32 buffers/accumulation. |
| TF32x3 tensor cores | Compile the matrix example with `--contraction-precision tf32x3`; inspect `mma.sync.aligned.m16n8k8` and the retained asynchronous copies. |
| Asynchronous copies | `cp.async` global-to-shared transfers of 4, 8, or 16 bytes, commit/wait groups, and barriers. Compile the matrix example with `--contraction-precision f32` for ordinary FP32 products and its three-stage copy ring. |
| FP16 storage | [Generated copy/conversion/arithmetic examples](examples/f16_storage.py). Scalar/vector half storage, exact widening, nearest-even narrowing, and explicitly typed FP32 computation. |
| Attention optimization | [Standalone attention example](examples/attention.md). Scores in registers, shared probabilities, and a guarded warp-distributed vector value reduction with correct in-place denominator replay. |
| Reduction and memory passes | Coalesced reductions, register forwarding, tree specialization, split-K folding, shared-bank layout, packed register/fragment staging, pipeline drain, shared-storage reuse, and CTA address factoring. |
| Packaging and validation | Installable `deuteron` command; a restricted constructor parser; option, launch, and ABI validation; bounded memory caching and atomic, integrity-checked persistent records. |
| Runtime and concurrency | Serialized LLVM calls within a process, spawned compilation workers, explicit local-context ownership, bounded remote retries, stale-cubin recovery, and output/timing checks. |

Compile and test from the repository root:

```sh
python -m pip install .
deuteron --version
deuteron examples/tensor_matmul/matmul_tiled_128.ir.json --arch sm_120 > /tmp/tensor.ptx
python examples/f16_storage.py --output /tmp/f16-examples
python examples/attention.py --output /tmp/attention-example
python -m unittest discover -s tests -v
DEUTERON_TEST_DEVICE=1 python -m unittest discover -s tests -v
```

The final command opts into numerical CUDA tests on the chosen available device. Setting `PTXAS` enables optional assembly checks. Tests include all half bit patterns, conversion-rounding boundaries, unaligned vector access, all tensor precision modes, partial final blocks, attention ownership/suffix replay, and invalid prefetch-phase rejection. The [validation, cache, runtime, and CLI tests](README.md#examples-and-tests) also cover constructor non-execution, malformed interfaces, corruption/concurrency, remote failures, and file preservation. Python 3.13 passes all 103 tests with CUDA/assembly enabled and no skips. Python 3.12 CPU discovery runs 103 tests with 11 expected skips and no failures. The original suite passes 15/15 at O3 and 15/15 at O2. Local suites use GPUs 1–7 with clean occupancy checks; GPU 0 is excluded. All 11 native SM120 RTX 5090 feature examples pass. Compilation itself needs no GPU or CUDA toolkit. Python 3.12 and 3.13 are the configured CPU CI versions.

## Performance evidence and scope

The [round-6 RTX 5090 remeasurement](benchmarks/rtx5090-2026-10-24-r6.json) gives geometric-mean warm latencies of **3.752 µs for Deuteron**, **6.216 µs for generated CUDA/NVCC**, and **4.332 µs for the tested Triton baseline** across the 15 open workloads: speedups of **1.66×** and **1.15×**. All 270 balanced-order warm calls pass the original output checks. See the [timing table](README.md#rtx-5090-performance), [release validation](benchmarks/validation-2026-10-24-r6.json), and [reproduction instructions](benchmarks/README.md). This round hardens packaging and interfaces while preserving the measured kernels; it does not establish a new optimization gain. The separate [single-launch profile](benchmarks/cold-rtx5090-2026-10-24-r6.json) retains all 90 correct calls at zero warmups and one iteration, with a geometric-mean latency of 19.648 µs and no comparative speedup claim.

Public storage supports FP32 and the documented FP16 subset. BF16/TF32 are internal tensor formats; native half arithmetic/atomics, half shared/tensor paths, public BF16 buffers, TMA, WGMMA, and TMEM are not implemented. Scalar kernel parameters, nonlinear launches, and serial/auxiliary thread axes are unsupported. Approximate math and compensated tensor modes are validated against the original tolerances, not claimed to be universally IEEE-equivalent. This is a scheduled-IR backend, not a general graph compiler or a complete IR safety verifier. The [deployment contract](docs/production.md) documents alignment, nonaliasing, synchronization, and service-isolation obligations. Hidden-set support is not claimed.

LLVM supplies optimization and code generation. Eligibility under the hackathon's broad prohibition on existing compilers remains unconfirmed.
