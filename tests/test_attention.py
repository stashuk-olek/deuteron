from collections import Counter
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from compiler import Options, compile_kernel
from compiler.codegen.lower import Lowering
from compiler.passes.recurrence import _grouped_value_axis, stage_recurrences
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign
from examples.attention import example, reference, write_example
from llvmlite import ir


def decode(graph, metadata):
    parsed = Graph.from_dict(graph)
    kernel = next(node.op for node in parsed.nodes.values() if type(node.op).__name__ == 'KernelOp')
    launch = metadata['kernels'][0]
    plan, = stage_recurrences(kernel, launch)
    return kernel, launch, plan, {arg['name']: tuple(arg['shape']) for arg in launch['args']}


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel, cls.launch, cls.plan, cls.shapes = decode(*example())

    def accepts(self, plan=None, launch=None, shapes=None):
        return _grouped_value_axis(plan or self.plan, launch or self.launch, shapes or self.shapes) is not None

    def test_example_compiles_and_owns_all_values(self):
        self.assertTrue(self.accepts())
        with tempfile.TemporaryDirectory() as directory:
            source, _ = write_example(directory)
            ptx = compile_kernel(source, opt_level=3).ptx
            self.assertIn('ld.global.nc.v4', ptx)
            self.assertIn('shfl.sync.bfly.b32', ptx)
        for keys in range(32, 257, 32):
            with self.subTest(keys=keys):
                counts = Counter((4 * item + lane // 8, 4 * (lane % 8) + component) for lane in range(32) for item in range(keys // 4) for component in range(4))
                self.assertEqual(counts, Counter((key, column) for key in range(keys) for column in range(32)))
        groups = [{lane // 8} for lane in range(32)]
        for distance in (16, 8):
            self.assertTrue(all(not (groups[lane] & groups[lane ^ distance]) for lane in range(32)))
            groups = [groups[lane] | groups[lane ^ distance] for lane in range(32)]
        self.assertTrue(all(group == set(range(4)) for group in groups))
        self.assertEqual(Counter(4 * lane + part for lane in range(8) for part in range(4)), Counter(range(32)))

    def test_inplace_denominator_is_accepted(self):
        _, launch, plan, shapes = decode(*example(inplace_denominator=True))
        self.assertTrue(self.accepts(plan, launch, shapes))
        self.assertEqual(plan.suffix[0].name, plan.components[0][2])

    def test_unclosed_suffix_declines_grouping(self):
        extra = Assign('unreplayed', ElementwiseImpl('copy'), (self.plan.components[0][1],))
        tile = replace(self.plan.tile, body=Body((*self.plan.prefix, self.plan.loop, extra, *self.plan.suffix)))
        kernel = replace(self.kernel, body=Body((tile,)))
        plan, = stage_recurrences(kernel, self.launch)
        self.assertFalse(self.accepts(plan))

    def test_control_aliases_decline_grouping(self):
        names = [axis.name for axis in self.plan.tile.axes]
        for name in (*names, self.plan.reductions[0].axis.name, 'lane', 'warp', 'threadIdx.x', 'blockIdx.x'):
            with self.subTest(name=name):
                outer = replace(self.plan.loop, axis=Axis(name, Dim(self.plan.extent)))
                self.assertFalse(self.accepts(replace(self.plan, loop=outer)))
        reduction = replace(self.plan.reductions[0], axis=Axis(names[0], Dim(32)))
        self.assertFalse(self.accepts(replace(self.plan, reductions=(reduction,))))

    def test_dependent_or_duplicate_output_rows_decline_grouping(self):
        for row in (Var('row') + Var('column') * 0, Literal(0, dtype='int')):
            with self.subTest(row=repr(row)):
                write = replace(self.plan.suffix[-1], index=(row, Var('column')))
                self.assertFalse(self.accepts(replace(self.plan, suffix=(*self.plan.suffix[:-1], write))))

    def test_mixed_storage_and_incomplete_warps_decline_grouping(self):
        args = [dict(arg, dtype='f16') if arg['name'] == 'v' else arg for arg in self.launch['args']]
        self.assertFalse(self.accepts(launch={**self.launch, 'args': args}))
        for block in ([16, 1, 1], [64, 1, 1]):
            self.assertFalse(self.accepts(launch={**self.launch, 'block': block}))
        self.assertFalse(self.accepts(launch={**self.launch, 'grid': [4, 1, 1]}))

    def test_post_tile_consumers_decline_grouping(self):
        kernel = replace(self.kernel, body=Body((self.plan.tile, self.plan.suffix[-1])))
        lowering = Lowering(kernel, self.launch, ir.Module(name='suffix_scope'), Options())
        self.assertTrue(lowering.recurrences)
        self.assertTrue(all(emitter.grouped_axis is None for emitter in lowering.recurrences.values()))

    def test_reference_uses_the_seeded_f32_inputs(self):
        _, metadata = example()
        arrays = [np.random.default_rng(arg['seed']).standard_normal(arg['shape']).astype(np.float32).astype(np.float64) for arg in metadata['kernels'][0]['args'][:-1]]
        scores = arrays[0] @ arrays[1] * np.float32(1 / np.sqrt(arrays[0].shape[1]))
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected = probabilities @ arrays[2] / probabilities.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(reference(metadata), expected, rtol=1e-12, atol=1e-12)


@unittest.skipUnless(os.environ.get('DEUTERON_TEST_DEVICE'), 'set DEUTERON_TEST_DEVICE to run CUDA correctness tests')
class AttentionCudaTests(unittest.TestCase):
    def test_all_key_groups_and_inplace_denominator(self):
        from benchmark import LocalRunner, compare_outputs

        with LocalRunner(int(os.environ['DEUTERON_TEST_DEVICE'])) as runner, tempfile.TemporaryDirectory() as directory:
            for keys in range(32, 257, 32):
                outputs = []
                for alias in (False, True):
                    with self.subTest(keys=keys, inplace_denominator=alias):
                        folder = Path(directory) / f'keys{keys}_alias{alias}'
                        source, metadata = write_example(folder, rows=1 + keys // 32, keys=keys, depth=32 if keys % 64 else 64, inplace_denominator=alias, seed=keys + 7)
                        compiled = compile_kernel(source, arch=runner.device_info['arch'], opt_level=3)
                        result = runner.run_kernel(metadata['kernels'][0], compiled.ptx, warmup=0, iters=1, samples=1)
                        compare_outputs(folder, metadata, result['outputs'])
                        outputs.append(result['outputs']['out'])
                np.testing.assert_array_equal(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
