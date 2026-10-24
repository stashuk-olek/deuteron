# Attention with a warp-distributed value reduction

This example computes `out = softmax((q @ kt) / sqrt(depth)) @ v` using f32
buffers and arithmetic. It generates its own IR, launch metadata and float64
NumPy reference from seeded f32 inputs. The default launch has three CTAs of
32 threads, 64 keys, a dot depth of 32 and 32 output columns.

The compiler recognizes the online recurrence, shares scores and the complete
softmax denominator across each warp, and distributes the final value
reduction across four key groups. Each lane loads four adjacent value columns;
warp shuffles combine the groups. This keeps all keys and outputs while using
vector loads. It does not use tensor cores or change buffer precision.

From the repository root:

```sh
python examples/attention.py --output /tmp/attention-example
python compile.py /tmp/attention-example/kernel.ir.json > /tmp/attention.ptx
python -m unittest discover -s tests -p test_attention.py -v
```

The CPU tests check ownership, guard rejection and compilation. To run the
numerical checks on an available GPU, select it explicitly:

```sh
python examples/attention.py --output /tmp/attention-alias --inplace-denominator --device 1
DEUTERON_TEST_DEVICE=1 python -m unittest discover -s tests -p test_attention.py -v
```

`--inplace-denominator` generates a legal suffix that writes the reciprocal
back into the denominator variable. Each output-column replay must restore
that variable before evaluating the suffix. The regression tests cover this
case, all supported key-group extents from 32 to 256, output ownership, and
rejection of suffixes or counter aliases that invalidate independent replay.
