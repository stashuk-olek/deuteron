# Deuteron submission

Deuteron lowers scheduled Emmy Kernel IR through custom GPU passes and LLVM NVPTX to executable PTX. Its focus is RTX 5090 performance while retaining the original launch, inputs, and numerical tolerances. See the [architecture and complete pass inventory](README.md#how-compilation-works).

## Features to inspect

| Feature | Where to see it |
|---|---|
| BF16x3 tensor cores | [128³ matrix example](examples/tensor_matmul/README.md): `cvt.rn.bf16x2`, `ldmatrix`, and `mma.sync.aligned.m16n8k16`, with FP32 buffers/accumulation. |
| TF32x3 tensor cores | Compile the matrix example with `--contraction-precision tf32x3`; inspect `mma.sync.aligned.m16n8k8` and the retained asynchronous copies. |
| Copy engine | `cp.async` global-to-shared transfers, commit/wait groups, and barriers. Compile the matrix example with `--contraction-precision f32` to inspect its three-stage copy ring. |
| FP16 storage | [Generated copy/conversion/arithmetic examples](examples/f16_storage.py). Scalar/vector half storage, exact widening, nearest-even narrowing, and explicitly typed FP32 computation. |
| Attention optimization | [Standalone attention example](examples/attention.md). Scores in registers, shared probabilities, and a guarded warp-distributed vector value reduction with correct in-place denominator replay. |
| Reduction and memory passes | Coalesced reductions, register forwarding, tree specialization, split-K folding, shared-bank layout, packed register/fragment staging, pipeline drain, shared-storage reuse, and CTA address factoring. |

Compile and test from the repository root:

```sh
python -m pip install -r requirements.txt
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json --arch sm_120 > /tmp/tensor.ptx
python examples/f16_storage.py --output /tmp/f16-examples
python examples/attention.py --output /tmp/attention-example
python -m unittest discover -s tests -v
DEUTERON_TEST_DEVICE=1 python -m unittest discover -s tests -v
```

The final command opts into numerical CUDA tests on the chosen available device. Setting `PTXAS` enables optional assembly checks. Tests include all half bit patterns, conversion-rounding boundaries, unaligned vector access, all tensor precision modes, partial final blocks, attention ownership/suffix replay, and invalid prefetch-phase rejection. Compilation itself needs no GPU or CUDA toolkit.

## Performance evidence and scope

On the 15 open workloads, Deuteron achieves a **1.66× speedup over generated CUDA/NVCC** and **1.15× over the tested Triton baseline** by geometric-mean warm latency. See the [RTX 5090 comparison](README.md#rtx-5090-performance) and [reproduction instructions and raw measurements](benchmarks/README.md). The comparison uses 270 correctness-checked warm calls with balanced order. It is separate from the cold-launch experiment for the latest pass, which reduced code/register footprint without a confirmed latency gain.

Public storage supports FP32 and the documented FP16 subset. BF16/TF32 are internal tensor formats; native half arithmetic, public BF16 buffers, TMA, WGMMA, and TMEM are not implemented. Approximate math and compensated tensor modes are validated against the original tolerances, not claimed to be universally IEEE-equivalent. Hidden-set support is not claimed.

LLVM supplies optimization and code generation. Eligibility under the hackathon's broad prohibition on existing compilers remains unconfirmed.
