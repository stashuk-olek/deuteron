# Deployment contract

Deuteron is a compiler for the documented subset of **scheduled Emmy Kernel IR**. Version 0.1.0 adds an installable command, restricted IR parsing, validated launch metadata, bounded process-local caching, atomic disk caching, and regression tests. These controls make the supported interface suitable for integration and repeatable evaluation. They do not establish support for arbitrary graphs or a hidden workload set.

## Installation and execution

Use a dedicated Python 3.12 or 3.13 environment and install the checkout with `python -m pip install .`. The `deuteron` command and `python compile.py` share the same options. Compilation requires Emmy and llvmlite; it does not initialize CUDA or require a GPU. The requirements file pins the dependency versions used for validation. NVIDIA assembly and execution are separate steps that require a compatible toolkit/driver and target GPU.

The CLI writes PTX to standard output and diagnostics to standard error. Invalid arguments, malformed input, unsupported features, and file errors produce a nonzero exit status. LLVM dumps use separate output paths; they must not overwrite the input IR, its adjacent metadata, or each other. Shell redirection is performed by the shell before the compiler starts, so keep redirected PTX paths separate from all inputs as well.

The Python interface remains:

```python
from pathlib import Path
from compiler import Options, clear_cache, compile_kernel, compile_many

source = Path("examples/vector_add/vector_add.ir.json")
result = compile_kernel(source, arch="sm_120", opt_level=3, options=Options())
Path("/tmp/vector_add.ptx").write_text(result.ptx)
clear_cache()
```

`Compilation` contains PTX, unoptimized LLVM IR, and optimized LLVM IR. CLI defaults remain SM89/O3; API defaults remain SM89/O2. Choose `sm_120` explicitly for RTX 5090. Changing compiler options creates a distinct cache entry. The API accepts nonnegative signed 32-bit unroll hints; the CLI offers the curated choices shown by `--help`.

## Input and kernel ABI

The IR file needs an adjacent `meta.json` with one matching launch. Compilation never reads expected output arrays, reference CUDA, or benchmark timing data. The restricted parser accepts known Emmy IR constructors and literal containers; it does not evaluate arbitrary Python. Unknown constructors, attribute access, comprehensions, and arbitrary calls are rejected.

Launch validation checks argument names/order, supported transport types, shapes, CUDA dimensions, shared-memory sizes, and output-reset names. The compiler then applies its narrower supported-kernel checks before optimization. Invalid API option values fail explicitly. A transform that cannot prove its own preconditions retains the existing generic path where that path supports the input.

Callers must provide:

- Distinct, nonoverlapping global buffer allocations, each with a base address aligned to at least 16 bytes. Full-FP32 vector accesses must also have naturally aligned offsets: 8 bytes for two values and 16 bytes for four. Generated LLVM uses `noalias` and alignment attributes; mixed FP16/FP32 storage uses conservative scalar access alignment.
- Buffer types, shapes, argument order, and launch dimensions matching metadata. The block size is fixed in emitted PTX with `.reqntid`.
- The declared shared-memory allocation and output resets. Atomic accumulation kernels require their output reset before each independent execution.
- A valid scheduled program: initialized values, correct synchronization and memory lifetimes, and in-bounds accesses on every executed path.

Validation is not a complete proof of arbitrary IR memory safety, race freedom, termination, or numerical equivalence. The compiler does not insert general bounds checks or change the supplied launch to repair an invalid schedule. For a service accepting untrusted input, use process isolation and enforce CPU, memory, input-size, and compilation-time limits outside the compiler. Restricted parsing removes Python expression execution; it is not an operating-system sandbox for LLVM or emitted kernels.

## Precision and hardware scope

The full scheduled backend uses FP32 storage. The documented pointwise subset also supports FP16 global storage with explicit FP32 arithmetic, exact widening, nearest-even narrowing, and bit-preserving raw copies. Public BF16 buffers, native half arithmetic/atomics, and half shared/tensor paths are unsupported.

BF16x3 and TF32x3 tensor products use FP32 inputs/accumulation and compensated internal operands. They remain approximations. `--no-approximate-math` disables selected reciprocal/sqrt/division approximations; exponentiation still uses approximate GPU instructions. The tests cover the supplied tolerance contracts and targeted conversion cases, not universal IEEE behavior for arbitrary exceptional values.

Supported emission targets are SM80, SM86, SM89, SM90, SM100, SM110, and SM120. Execution is validated on H100 and RTX 5090. SM110 uses PTX 9.0 and is covered by compilation tests; this environment does not validate its assembly or execution. Tensor/copy support consists of `mma.sync`, `ldmatrix`, packed operands, and `cp.async` with commit/wait groups and barriers. TMA, WGMMA, and TMEM are not implemented.

## Caching and concurrency

Compilation is thread-safe within one process; LLVM work is serialized there. Use `compile_many(jobs, workers=N)` for process parallelism. Each job is `(path, arch, opt_level, options)`. Its spawn-based workers require an importable Python entry module and the usual `if __name__ == "__main__":` guard. Results retain input order; failures propagate to the caller.

The process-local cache holds at most 128 entries and 64 MiB of cached text. `clear_cache()` clears cached compilations; it leaves disk files and loaded modules intact. Restart the process after changing compiler source. Set `DEUTERON_CACHE` to enable persistent caching. Cache identity includes compiler source/dependency versions, input IR, metadata, target, optimization level, and options. Records have a format version and content-integrity checks. Missing, unreadable, or malformed records are misses; failed writes do not invalidate a successful compilation. Writers use distinct temporary files and atomic replacement.

The persistent cache is an optional acceleration layer, not an artifact registry. Its disk size is managed by the deployer, and its contents should be owned by the same trusted user as the compiler. Integrity hashes detect corruption; they do not authenticate records from an untrusted writer. Pin versions and record emitted PTX hashes when distributing compiled artifacts.

## Runtime and measurements

`benchmark.py` is a validation/measurement utility, not a general tensor runtime. A `LocalRunner` owns a CUDA context and must be used and closed by the process and thread that created it. Context, module, graph, event, and allocation cleanup occurs on normal and exceptional exits. Explicitly select idle GPUs when sharing a machine.

The remote client preserves launch/reset metadata, bounds busy retries, caches assembly records, and recompiles once when the server explicitly reports an unknown or expired cubin. Other execution or transport errors are reported without automatic execution replay. It checks output names, shapes, truncation, and timing validity; the comparison harness then checks complete values against the original tolerances. Remote `samples` does not control server-side sampling; the returned latency is the server's `median_us` for the requested warmup/iteration counts.

Output shapes must match metadata and reference arrays. The original reduction references use NumPy scalars for buffers declared with shape `[1]`; the comparison utility accepts that specific representation after checking the produced buffer's declared shape. It does not silently flatten arbitrary mismatched shapes.

Benchmark JSON updates use atomic replacement. The benchmark CLI requires a new artifact directory for each run, preserving earlier manifests and measurements. Reports record the complete compiler source hashes. Warm kernel latency excludes compilation, assembly, and transport; it is not end-to-end application latency. See [measurement protocol and retained results](../benchmarks/README.md).

## Validation before release

The repository's CPU CI matrix runs the public tests on Python 3.12/3.13, exercises all seven emission targets, checks types, and builds/installs the package. CUDA tests are opt-in through `DEUTERON_TEST_DEVICE`; `PTXAS` enables assembly checks. A CPU CI pass does not imply numerical GPU validation.

For a compiler change, verify the documented examples and original workloads at the affected options/targets, compare emitted PTX and resource use, and measure on the target GPU. Preserve failed and contaminated attempts. The committed validation records identify exactly which checks were run; adding a workflow does not claim that a remote CI service has executed it.
