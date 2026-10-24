# Tiled matrix multiplication with tensor cores and asynchronous copies

This self-contained example computes `result = left @ right` for two 128 × 128
FP32 matrices. The launch uses eight CTAs of 512 threads. Each CTA computes a
32 × 64 output tile and advances through K in two tiles of 64 values.

The scheduled IR contains a three-stage shared-memory ring with two stages
prefetched ahead. Sixteen-byte `cp.async` copies, commit/wait groups, and CTA
barriers make the transfer and consumption order explicit. Its final prefetch
coordinates clamp to K = 64, so every copy remains inside its input buffer.

The default compiler proves this schedule can use packed BF16 operands. It
splits each FP32 operand into a rounded high part and a rounded residual, and
accumulates three tensor products: high × high, high × low, and low × high.
Packing occurs inside the kernel; the input and output buffers remain FP32.
The emitted PTX uses `cvt.rn.bf16x2`, `ldmatrix`, and
`mma.sync.aligned.m16n8k16`. This path replaces the asynchronous ring with
register prefetching and packed shared-memory staging.

Choose `f32` to use ordinary FP32 arithmetic and retain the IR's `cp.async`
ring. Choose `tf32x3` for three TF32 tensor products with FP32 accumulation;
this path also retains the asynchronous copies. The tensor modes approximate
FP32 multiplication and are checked against the example's `rtol=atol=0.001`.

From the repository root:

```sh
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json > /tmp/tensor.ptx
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json --contraction-precision f32 > /tmp/copy.ptx
python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json --arch sm_120 > /tmp/tensor-sm120.ptx
PTXAS=/usr/local/cuda/bin/ptxas python -m unittest discover -s tests -p test_tensor_copy.py -v
```

Compilation and optional assembly tests do not require a GPU. Assembly for
SM120 requires CUDA 12.8 or newer. To run numerical checks, select an available
GPU explicitly:

```sh
DEUTERON_TEST_DEVICE=1 python -m unittest discover -s tests -p test_tensor_copy.py -v
python benchmark.py tensor_matmul --problems-dir examples --devices 1 --warmup 1 --iters 1 --samples 1 --artifacts /tmp/tensor-example-results
```

`meta.json` fixes seeds 7 and 11. `expected.npz` contains a float64 matrix
product of the generated FP32 inputs. The tests verify that reference, compile
all three precision modes, and check that an incorrect prefetch phase is
rejected by register packing. No external benchmark files are needed.
