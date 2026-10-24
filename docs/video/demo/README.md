# CPU-only tensor and copy demo

From a checkout with Deuteron's dependencies installed in the active Python environment:

```sh
python docs/video/demo/run.py --output /tmp/deuteron-video-demo
```

Choose a new output directory for every recording. The script refuses an existing directory, uses the current Python interpreter, clears `PYTHONPATH` and compiler caching, and hides CUDA devices. It displays the real compile command followed by the first emitted `mma.sync` instruction and the first global-to-shared `cp.async` instruction. It saves PTX, the exact command/transcript, compiler diagnostics, and provenance in the output directory.

The checked-in [capture](capture/transcript.txt) compiles the independent 128 × 128 matrix example using SM120/O3 and `tf32x3`. Its compiler source matches release `efb728d9106d864bb5f35efc95bfb5f01d95646a`. [Provenance](capture/provenance.json) records every compiler-file hash, input hashes, PTX hash, and exit code 0.

Actual instruction lines in [kernel.ptx](capture/kernel.ptx):

```ptx
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%r27,%r28,%r29,%r30}, {%r21,%r22,%r23,%r24}, {%r25,%r26}, {%r27,%r28,%r29,%r30};
cp.async.cg.shared.global [%r1], [%rd1], 16;
```

This capture performs compilation and text inspection only. It performs no assembly, GPU execution, numerical check, or benchmark. Its emitted PTX hash matches the earlier validated `tensor_tf32x3` example in [the release feature record](../../../benchmarks/features-rtx5090-2026-10-24-r6.json). Presentation performance claims come from [claims.json](../claims.json) and its referenced measurement records.
