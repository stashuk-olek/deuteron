import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compiler import compile_kernel
from emmy.compiler.dtype import F16, F32
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.leaves import Assign
from examples.f16_storage import example


def tile_of(graph):
    parsed = Graph.from_dict(graph)
    return next(n.op.body[0] for n in parsed.nodes.values() if type(n.op).__name__ == 'KernelOp')


def with_body(graph, body):
    result = copy.deepcopy(graph)
    result['nodes']['result']['op_fields']['body'] = [repr(replace(tile_of(graph), body=Body(body)))]
    return result


def compile_example(graph, meta):
    with tempfile.TemporaryDirectory(prefix='deuteron-f16-') as directory:
        source = Path(directory) / 'kernel.ir.json'
        source.write_text(json.dumps(graph))
        source.with_name('meta.json').write_text(json.dumps(meta))
        return compile_kernel(source, opt_level=3)


class StorageTypingTests(unittest.TestCase):
    def test_explicit_float_compute_and_half_store(self):
        graph, meta = example('affine')
        compiled = compile_example(graph, meta)
        self.assertIn('fpext half', compiled.unoptimized)
        self.assertIn('fptrunc float', compiled.unoptimized)

    def test_missing_load_stamp_uses_buffer_dtype(self):
        graph, meta = example('copy')
        tile = tile_of(graph)
        graph = with_body(graph, (replace(tile.body[0], dtype=None), *tile.body[1:]))
        self.assertIn('half', compile_example(graph, meta).unoptimized)

    def test_mismatched_load_dtype(self):
        graph, meta = example('copy')
        tile = tile_of(graph)
        graph = with_body(graph, (replace(tile.body[0], dtype=F32), *tile.body[1:]))
        with self.assertRaisesRegex(ValueError, 'Load dtype disagrees'):
            compile_example(graph, meta)

    def test_mismatched_store_value_dtype(self):
        graph, meta = example('affine')
        tile = tile_of(graph)
        graph = with_body(graph, (*tile.body[:-1], replace(tile.body[-1], value_dtype=F16)))
        with self.assertRaisesRegex(ValueError, 'Write value dtype disagrees'):
            compile_example(graph, meta)

    def test_native_and_implicit_half_arithmetic_rejected(self):
        for dtype in (F16, None):
            with self.subTest(dtype=dtype):
                graph, meta = example('widen')
                tile = tile_of(graph)
                arithmetic = Assign(tile.body[1].name, 'add', (tile.body[0].names[0], tile.body[0].names[0]), dtype)
                graph = with_body(graph, (tile.body[0], arithmetic, *tile.body[2:]))
                with self.assertRaisesRegex(NotImplementedError, 'explicit f32 result'):
                    compile_example(graph, meta)

    def test_non_pointwise_storage_rejected(self):
        graph, meta = example('copy')
        graph = with_body(graph, (*tile_of(graph).body, Sync()))
        with self.assertRaisesRegex(NotImplementedError, 'f16 storage does not support Sync'):
            compile_example(graph, meta)

    def test_atomic_half_store_rejected(self):
        graph, meta = example('copy', width=1)
        tile = tile_of(graph)
        graph = with_body(graph, (*tile.body[:-1], replace(tile.body[-1], atomic=True)))
        with self.assertRaisesRegex(NotImplementedError, 'ordinary global stores'):
            compile_example(graph, meta)

    def test_store_to_readonly_input_rejected(self):
        graph, meta = example('copy')
        tile = tile_of(graph)
        graph = with_body(graph, (*tile.body[:-1], replace(tile.body[-1], output='x')))
        with self.assertRaisesRegex(NotImplementedError, 'ordinary global stores'):
            compile_example(graph, meta)


@unittest.skipUnless('DEUTERON_TEST_DEVICE' in os.environ, 'set DEUTERON_TEST_DEVICE to run CUDA correctness tests')
class StorageCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmark import LocalRunner
        cls.runner = LocalRunner(int(os.environ['DEUTERON_TEST_DEVICE']))

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()

    def run_values(self, graph, meta, values):
        compiled = compile_example(graph, meta)
        launch = copy.deepcopy(meta['kernels'][0])
        for arg in launch['args']:
            if arg['name'] in values:
                arg.pop('seed', None)
                arg['data'] = values[arg['name']]
        return self.runner.run_kernel(launch, compiled.ptx, warmup=0, iters=1, samples=1)['outputs']['result']

    def test_all_half_bits_copy_exactly_at_scalar_and_vector_widths(self):
        data = np.arange(65536, dtype=np.uint16).view(np.float16)
        for width in (1, 2, 4):
            with self.subTest(width=width):
                graph, meta = example('copy', count=data.size, width=width)
                actual = self.run_values(graph, meta, {'x': data})
                np.testing.assert_array_equal(actual.view(np.uint16), data.view(np.uint16))

    def test_half_widening(self):
        data = np.arange(65536, dtype=np.uint16).view(np.float16)
        graph, meta = example('widen', count=data.size)
        actual = self.run_values(graph, meta, {'x': data})
        expected = data.astype(np.float32)
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))

    def test_round_to_nearest_even_half_boundaries(self):
        values = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float32)
        mid = (values[:-1] + values[1:]) * np.float32(0.5)
        points = np.concatenate((np.nextafter(mid, -np.inf), mid, np.nextafter(mid, np.inf)))
        points = np.concatenate((points, -points, np.array([0., -0., 65504., 65520., 65519.996, np.inf, -np.inf, np.nan], dtype=np.float32)))
        points = np.pad(points, (0, -points.size % 4))
        graph, meta = example('narrow', count=points.size)
        actual = self.run_values(graph, meta, {'x': points})
        with np.errstate(over='ignore', invalid='ignore'):
            expected = points.astype(np.float16)
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[finite].view(np.uint16), expected[finite].view(np.uint16))
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))

    def test_explicit_float_compute_avoids_half_intermediate_rounding(self):
        graph, meta = example('cancellation')
        x = np.resize(np.array([2048., 4096., 65504., -2048.], dtype=np.float16), 512)
        y = np.ones(512, dtype=np.float16)
        actual = self.run_values(graph, meta, {'x': x, 'y': y})
        np.testing.assert_array_equal(actual, y)

    def test_unaligned_vector_half_access(self):
        for kind in ('copy', 'narrow'):
            with self.subTest(kind=kind):
                graph, meta = example(kind, count=512, offset=1)
                data = np.arange(513, dtype=np.float16 if kind == 'copy' else np.float32)
                actual = self.run_values(graph, meta, {'x': data})
                np.testing.assert_array_equal(actual, data[1:].astype(np.float16))
        graph, meta = example('widen', count=512)
        tile = tile_of(graph)
        graph = with_body(graph, (*tile.body[:-1], replace(tile.body[-1], index=(tile.body[-1].index[0] + 1,))))
        graph['nodes']['result']['outputs'][0]['shape'] = [513]
        meta['kernels'][0]['args'][-1]['shape'] = [513]
        data = np.arange(512, dtype=np.float16)
        actual = self.run_values(graph, meta, {'x': data})
        np.testing.assert_array_equal(actual[1:], data.astype(np.float32))

    def test_strided_loop_vectorization_preserves_half_storage(self):
        graph, meta = example('widen', count=512, width=1)
        tile = tile_of(graph)
        body = tuple(replace(s, index=(Var('item'),)) if type(s).__name__ in ('Load', 'Write') else s for s in tile.body)
        loop = StridedLoop(axis=Axis(name='item', extent=Dim(512)), start=Var('element'), step=Literal(128, dtype='int'), body=Body(body))
        tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(128)),), body=Body((loop,)))
        graph['nodes']['result']['op_fields']['body'] = [repr(tile)]
        meta['kernels'][0]['grid'][0] = 1
        compiled = compile_example(graph, meta)
        self.assertIn('<4 x half>', compiled.unoptimized)
        data = np.arange(512, dtype=np.float16)
        actual = self.run_values(graph, meta, {'x': data})
        np.testing.assert_array_equal(actual, data.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
