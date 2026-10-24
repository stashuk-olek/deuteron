# RTX 5090 compiler comparison

The [recorded results](rtx5090-2026-10-24.json) compare three implementations of the 15 open Emmy workloads on the hosted RTX 5090. The main [README](../README.md#rtx-5090-performance) contains the timing table and interpretation.

| Implementation | Compilation policy |
|---|---|
| Deuteron | Native SM120, O3, default compiler options; Emmy 0.3.11, llvmlite 0.50.0 / LLVM 22.1.0. Original launch and reset metadata. |
| NVCC 12.8.93 | Original Emmy-generated `.ref.cu`, `-O3 --std=c++17 -arch=compute_120 --ptx`, default math and FMA. Original launch and reset metadata. |
| Triton 3.6.0 | [Independent kernels](triton_kernels.py), native CUDA target 120, PTX 8.7, default FP fusion, BF16x3 dot products. Own tiling, launch geometry, and scratch ABI. |

All inputs, scalar-buffer constants, output shapes, references, and tolerances are unchanged. Every measurement validates the full output. Deuteron and NVCC consume the supplied schedule; Triton implements the same mathematical workloads with its own schedule. NVCC uses ordinary FP32 products; the tensor implementations use their documented approximations. Deuteron attention remains FP32, while Triton attention uses BF16x3 dot products. The rectangular matmul is an atomic split-K reduction in Deuteron/NVCC and a direct overwrite in Triton, so only the first two require per-iteration output resets.

These are warm hosted-runner kernel timings: 20 warmups and 200 iterations per call, six calls per implementation/workload, 270 calls total. Each problem uses all six compiler-order permutations once; problem order is shuffled independently in each round. The schedule is saved before execution. Reported values are medians of the six call medians; every sample is retained. Compilation, assembly, network transfer, and configuration-screen timings are excluded. No claim is made about the competition's cold-launch score, end-to-end model speed, or statistical significance of tiny timing differences.

The geometric mean gives each of the 15 workloads equal weight. Relative speed is the ratio of geometric-mean latencies, equivalently the geometric mean of per-workload speedups. These small workloads do not measure peak device throughput.

The fixed Triton screen includes:

- One configuration each for pointwise kernels, sum/mean, and row normalization/reduction.
- Matmul tiles `(BM, BN, BK)` of `(32,64,32)`, `(64,64,32)`, and `(32,64,64)`, four warps, three stages.
- Attention `(BM, warps)` of `(16,4)`, `(32,4)`, and `(16,8)`, two stages.

Attention/matmul candidates receive two screen calls; other kernels receive one. Selection uses the lowest median screen latency, breaking ties by configuration index. The selected configuration is frozen before the independent final run. This is a limited reproducible baseline, not exhaustive tuning or a claim to outperform every Triton implementation. cuBLAS, cuDNN, and complete PyTorch models are not measured.

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

The comparison tools default to the hosted runner; `--server` selects another compatible endpoint. `--lock` lets concurrent tools share one serialization lock. Use new output locations to preserve previous measurements. A new run uses a fresh saved schedule; `--seed` reproduces order when compiler/problem sets are unchanged.

`prepare.py` saves PTX, compiler/source hashes, compile options, and exact launch metadata. Triton's two unused scratch-pointer parameters receive dummy buffers only after the compiler reports zero scratch requirements. Its native PTX is assembled as `sm_120a`; the tool does not rewrite the target directive. The baseline kernels accept FP32 buffers only.

The result JSON contains all 270 calls, selected variants and launch records, source/PTX/reference hashes, numerical error checks, and the saved schedule. Its final-source verification confirms that the later SM110 target-version correction leaves all measured SM120 PTXs unchanged. [Configuration-screen evidence](rtx5090-screen-2026-10-24.json) records all 23 candidates. [Attention-pass confirmation](attention-confirmation-2026-10-24.json) separately records the fixed 100-call cold experiment; its result is a footprint improvement without a confirmed latency gain. [Native RTX 5090 feature checks](features-rtx5090-2026-10-24.json) record 11 passing runs of the new examples, including all three tensor/copy modes and half storage.
