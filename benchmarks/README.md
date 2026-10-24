# RTX 5090 compiler comparison

The [round-6 recorded results](rtx5090-2026-10-24-r6.json) compare three implementations of the 15 open Emmy workloads on the hosted RTX 5090. The main [README](../README.md#rtx-5090-performance) contains the per-workload timing table and interpretation. The [previous comparison](rtx5090-2026-10-24.json) remains available as historical evidence.

| Implementation | Warm geometric-mean latency | Deuteron speedup |
|---|---:|---:|
| Deuteron | 3.752 µs | 1.00× |
| NVCC 12.8.93 | 6.216 µs | 1.66× |
| Triton 3.6.0 | 4.332 µs | 1.15× |

| Implementation | Compilation policy |
|---|---|
| Deuteron | Native SM120, O3, default compiler options; Emmy 0.3.11, llvmlite 0.50.0 / LLVM 22.1.0. Original launch and reset metadata. |
| NVCC 12.8.93 | Original Emmy-generated `.ref.cu`, `-O3 --std=c++17 -arch=compute_120 --ptx`, default math and FMA. Original launch and reset metadata. |
| Triton 3.6.0 | [Independent kernels](triton_kernels.py), native CUDA target 120, PTX 8.7, default FP fusion, BF16x3 dot products. Own tiling, launch geometry, and scratch ABI. |

All inputs, scalar-buffer constants, output shapes, references, and tolerances are unchanged. Every measurement validates the full output. Deuteron and NVCC consume the supplied schedule; Triton implements the same mathematical workloads with its own schedule. NVCC uses ordinary FP32 products; the tensor implementations use their documented approximations. Deuteron attention remains FP32, while Triton attention uses BF16x3 dot products. The rectangular matmul is an atomic split-K reduction in Deuteron/NVCC and a direct overwrite in Triton, so only the first two require per-iteration output resets.

These are warm hosted-runner kernel timings: 20 warmups and 200 iterations per call, six calls per implementation/workload, 270 calls total. All 270 calls pass the original output checks. Each problem uses all six compiler-order permutations once; problem order is shuffled independently in each round. The schedule is saved before execution. Reported values are medians of the six call medians; every sample is retained. Compilation, assembly, network transfer, and configuration-screen timings are excluded. No claim is made about the competition's cold-launch score, end-to-end model speed, or statistical significance of tiny timing differences.

The geometric mean gives each of the 15 workloads equal weight. Relative speed is the ratio of geometric-mean latencies, equivalently the geometric mean of per-workload speedups. These small workloads do not measure peak device throughput.

The fixed Triton screen includes:

- One configuration each for pointwise kernels, sum/mean, and row normalization/reduction.
- Matmul tiles `(BM, BN, BK)` of `(32,64,32)`, `(64,64,32)`, and `(32,64,64)`, four warps, three stages.
- Attention `(BM, warps)` of `(16,4)`, `(32,4)`, and `(16,8)`, two stages.

Attention/matmul candidates receive two screen calls; other kernels receive one. Selection uses the lowest median screen latency, breaking ties by configuration index. The selected configuration is frozen before the independent final run. Round 6 reuses those selections and the exact previously measured PTX, with a fresh balanced schedule and no additional tuning. Fresh NVCC builds match the frozen PTX; fresh Triton builds differ only in trailing source-path/debug sections, so the frozen artifacts remove that variation. Parser, cache, packaging, and runtime hardening preserve the generated source kernels; differences between timing rounds are not new optimization gains. This is a limited reproducible baseline, not exhaustive tuning or a claim to outperform every Triton implementation. cuBLAS, cuDNN, and complete PyTorch models are not measured.

## Reproduce

Use the original Emmy problem checkout, Deuteron's requirements, CUDA 12.8 NVCC, and a Python environment with `triton==3.6.0`. Triton is optional for Deuteron itself. Its preparation command can run in a separate environment; it uses an explicit target and does not initialize a GPU.

```sh
PROBLEMS=/path/to/emmy-hackathon-2026/problems
OUT=/tmp/deuteron-comparison

python benchmarks/prepare.py deuteron --problems-dir "$PROBLEMS" --output "$OUT/deuteron"
python benchmarks/prepare.py nvcc --problems-dir "$PROBLEMS" --output "$OUT/nvcc" --nvcc /usr/local/cuda/bin/nvcc
/path/to/triton-env/bin/python benchmarks/prepare.py triton --problems-dir "$PROBLEMS" --output "$OUT/triton"

python benchmarks/compare.py screen --problems-dir "$PROBLEMS" \
  --manifest "$OUT/triton/manifest.json" --out "$OUT/screen.json"
python benchmarks/compare.py run --problems-dir "$PROBLEMS" \
  --manifest "$OUT/deuteron/manifest.json" \
  --manifest "$OUT/nvcc/manifest.json" \
  --manifest "$OUT/screen-selected.json" --out "$OUT/results.json"
```

The comparison tools default to the hosted runner; `--server` selects another compatible endpoint. `--lock` lets concurrent tools share one serialization lock. Use new output locations to preserve previous measurements. A new run uses a fresh saved schedule; `--seed` reproduces order when compiler/problem sets are unchanged. Reports are written atomically. Runtime input/response validation, retry behavior, local context ownership, and caller obligations are described in the [deployment contract](../docs/production.md#runtime-and-measurements).

`prepare.py` saves PTX, compiler/source hashes, compile options, and exact launch metadata. Triton's two unused scratch-pointer parameters receive dummy buffers only after the compiler reports zero scratch requirements. Its native PTX is assembled as `sm_120a`; the tool does not rewrite the target directive. The baseline kernels accept FP32 buffers only.

The round-6 result JSON contains all 270 calls, selected variants and launch records, source/PTX/reference hashes, numerical error checks, the saved schedule, and an independent audit of the frozen artifacts. The warm run used the frozen previous harness and exact PTX. A fresh final-source audit reproduces all 45 PTX files and 90 LLVM stages and maps every measured Deuteron variant to the release compiler. The hardened runtime is separately exercised by the local suites, native examples, and single-launch profile. These measurements do not claim faster compilation or runtime dispatch. [Configuration-screen evidence](rtx5090-screen-2026-10-24.json) preserves all 23 original candidates.

The [historical attention-pass confirmation](attention-confirmation-2026-10-24.json) records the round-5 fixed 100-call experiment: code/register footprint improved, with no confirmed warm or cold latency gain. The separate [round-6 single-launch profile](cold-rtx5090-2026-10-24-r6.json) records six randomized rounds over all 15 workloads: 90 calls at zero warmups and one iteration, using default CLI SM89/O3 PTX assembled for SM120. All outputs pass; the geometric mean of the 15 medians is 19.648 µs. It retains the complete schedule and all samples, with no comparative speedup or significance claim. The first attempt stopped on the original scalar-reference file format; the corrected harness repeated the same schedule in full, preserving the failed attempt separately. No inputs or tolerances changed.

[Fresh native RTX 5090 feature checks](features-rtx5090-2026-10-24-r6.json) preserve 11 passing example runs, including BF16x3/TF32x3/F32 matrix modes, asynchronous copies, and half storage. See the [public test inventory](../README.md#examples-and-tests) and [release validation record](validation-2026-10-24-r6.json) for the expanded suite.

Final release checks pass all 103 Python 3.13 tests with CUDA/assembly enabled and no skips. Python 3.12 CPU discovery runs 103 tests with 11 expected skips and no failures. Original workloads pass 15/15 at O3 and 15/15 at O2; local suites use GPUs 1–7 with clean occupancy checks and exclude GPU 0. All 11 native SM120 RTX 5090 feature examples pass again. These correctness checks are separate from the timing profiles.
