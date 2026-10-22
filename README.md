# Deuteron

Compiles scheduled Emmy Kernel IR to LLVM IR and emits PTX with LLVM's NVPTX backend. Validated on all 15 open Emmy hackathon problems using H100 and RTX 5090 GPUs. Compilation needs no GPU, CUDA toolkit, PyTorch or Triton.

Requires Python 3.12 or newer:

```bash
python -m pip install -r requirements.txt
python compile.py /path/to/kernel.ir.json > kernel.ptx
```

The input's adjacent `meta.json` defines the argument order and launch geometry. The compiler reads neither expected outputs nor reference implementations.

Keep both LLVM stages when needed:

```bash
python compile.py /path/to/kernel.ir.json --dump-llvm kernel.ll --dump-unoptimized kernel.unoptimized.ll > kernel.ptx
```

Benchmark and validate against an external checkout of the hackathon problems:

```bash
python benchmark.py --problems-dir /path/to/emmy-hackathon-2026/problems --devices 0,1,2,3,4,5,6,7
python benchmark.py --problems-dir /path/to/emmy-hackathon-2026/problems --remote
```

Local benchmarking needs an NVIDIA GPU and CUDA driver. Remote benchmarking needs `curl` and defaults to the hosted RTX 5090 grader. Add problem names to select a subset. Results include timings, source hashes, PTX, both LLVM stages, and actual GPU outputs in `benchmark-results/`.

The pipeline runs structural GPU passes over Emmy IR first (affine indexing, reduction sharing and vectorization, loop transforms, online-recurrence staging, contraction tensorization, shared-memory layout), lowers the result to LLVM IR, then runs LLVM optimization plus custom LLVM cleanup before NVPTX emission. The default contraction policy uses three compensated BF16 tensor-core products with f32 accumulators for any eligible contraction when structural and hardware-utilization guards pass. `--contraction-precision f32` keeps scalar products; `tf32x3` is also available. `--no-approximate-math` restores the original reciprocal/sqrt lowering. Approximate modes are validated for the open inputs; they do not promise IEEE equivalence for infinities, overflow or subnormals. Precision options never change the benchmark's tolerances.

The compiler reuses LLVM for optimization and code generation. Eligibility under the hackathon's broad prohibition on existing compilers remains unconfirmed. The validated scope is the 15 open problems; hidden-set support is not claimed.
