from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from compiler import Options, compile_kernel
from compiler.analysis import kind
from compiler.passes.contraction import tensorize
from compiler.passes.regstage import stage_register_loads
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.body import Body


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'tensor_matmul'
IR = EXAMPLE / 'matmul_tiled_128.ir.json'
PTXAS = shutil.which(os.environ.get('PTXAS', 'ptxas'))
DEVICE = os.environ.get('DEUTERON_TEST_DEVICE')


class TensorCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((EXAMPLE / 'meta.json').read_text())
        cls.launch = cls.metadata['kernels'][0]
        cls.graph = Graph.from_dict(json.loads(IR.read_text()))
        cls.kernel = next(n.op for n in cls.graph.nodes.values() if kind(n.op) == 'KernelOp')

    def ptx(self, mode='bf16x3', arch='sm_89'):
        return compile_kernel(IR, arch=arch, opt_level=3, options=Options(contraction_precision=mode)).ptx

    def test_example_shapes_and_reference(self):
        raw = json.loads(IR.read_text())
        self.assertEqual(self.launch['grid'], [8, 1, 1])
        self.assertEqual(self.launch['block'], [512, 1, 1])
        self.assertEqual(self.launch['entry'], self.kernel.name)
        for node in raw['nodes'].values():
            for output in node['outputs']:
                self.assertEqual(output['shape'], [128, 128])
        arrays = {}
        for arg in self.launch['args']:
            self.assertEqual(arg['shape'], [128, 128])
            self.assertEqual(arg['dtype'], 'f32')
            if not arg.get('output'):
                arrays[arg['name']] = np.random.default_rng(arg['seed']).standard_normal(arg['shape']).astype(np.float32)
        reference = arrays['left'].astype(np.float64) @ arrays['right'].astype(np.float64)
        with np.load(EXAMPLE / 'expected.npz') as expected:
            self.assertEqual(expected.files, ['result'])
            np.testing.assert_allclose(expected['result'], reference, rtol=1e-12, atol=1e-12)

    def test_default_uses_packed_bf16_tensor_cores(self):
        self.assertIsNotNone(stage_register_loads(self.kernel, self.launch))
        ptx = self.ptx()
        self.assertIn('mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32', ptx)
        self.assertIn('cvt.rn.bf16x2.f32', ptx)
        self.assertIn('ldmatrix.sync.aligned.m8n8.x4.shared.b16', ptx)
        self.assertIn('ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16', ptx)

    def test_tf32_uses_tensor_cores_and_async_copies(self):
        ptx = self.ptx('tf32x3')
        self.assertIn('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32', ptx)
        self.assertIn('cvt.rna.tf32.f32', ptx)
        self.assertIn('cp.async.cg.shared.global', ptx)
        self.assertIn('cp.async.commit_group', ptx)
        self.assertIn('cp.async.wait_group 2', ptx)

    def test_f32_retains_copy_engine_and_barriers(self):
        ptx = self.ptx('f32')
        self.assertNotIn('mma.sync', ptx)
        self.assertIn('cp.async.cg.shared.global', ptx)
        self.assertIn('cp.async.commit_group', ptx)
        self.assertIn('cp.async.wait_group 2', ptx)
        self.assertIn('bar.sync', ptx)

    def test_wrong_prefetch_phase_declines_register_packing(self):
        plan = tensorize(self.kernel, self.launch, 'bf16x3')[0]
        extent = 192
        next_tile = Var(plan.outer.axis.name) + 2 * plan.depth
        future = TernaryExpr(BinaryExpr('<', next_tile, Literal(extent, dtype='int')), next_tile, Literal(extent - plan.depth, dtype='int'))

        def extend(expr):
            if kind(expr) == 'TernaryExpr':
                return future
            if kind(expr) == 'BinaryExpr':
                return replace(expr, left=extend(expr.left), right=extend(expr.right))
            return expr

        body = Body(tuple(replace(s, src_index=tuple(extend(e) for e in s.src_index)) if kind(s) == 'CpAsyncCopy' else s for s in plan.outer.body))
        outer = replace(plan.outer, axis=replace(plan.outer.axis, extent=Dim(extent)), body=body)
        tile = replace(plan.tile, body=Body(tuple(outer if s is plan.outer else s for s in plan.tile.body)))
        kernel = replace(self.kernel, body=Body((tile,)))
        launch = deepcopy(self.launch)
        for arg in launch['args']:
            if arg['name'] == 'left':
                arg['shape'][1] = extent
            elif arg['name'] == 'right':
                arg['shape'][0] = extent
        self.assertIsNotNone(stage_register_loads(kernel, launch))
        plan = tensorize(kernel, launch, 'bf16x3')[0]
        for canceled in (False, True):
            with self.subTest(canceled=canceled):
                changed = 0

                def rewrite(expr):
                    nonlocal changed
                    if kind(expr) == 'TernaryExpr':
                        changed += 1
                        current = Var(plan.outer.axis.name)
                        return current + expr - expr if canceled else current
                    if kind(expr) == 'BinaryExpr':
                        return replace(expr, left=rewrite(expr.left), right=rewrite(expr.right))
                    return expr

                body = Body(tuple(replace(s, src_index=tuple(rewrite(e) for e in s.src_index)) if kind(s) == 'CpAsyncCopy' else s for s in plan.outer.body))
                outer = replace(plan.outer, body=body)
                tile = replace(plan.tile, body=Body(tuple(outer if s is plan.outer else s for s in plan.tile.body)))
                altered = replace(kernel, body=Body((tile,)))
                self.assertGreater(changed, 0)
                self.assertTrue(tensorize(altered, launch, 'bf16x3'))
                self.assertIsNone(stage_register_loads(altered, launch))

    @unittest.skipUnless(PTXAS, 'set PTXAS or put ptxas on PATH for assembly checks')
    def test_modes_assemble(self):
        for mode in ('bf16x3', 'tf32x3', 'f32'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'kernel.ptx'
                cubin = Path(directory) / 'kernel.cubin'
                source.write_text(self.ptx(mode))
                result = subprocess.run([PTXAS, '-arch=sm_89', str(source), '-o', str(cubin)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertGreater(cubin.stat().st_size, 0)

    @unittest.skipUnless(PTXAS, 'set PTXAS or put ptxas on PATH for assembly checks')
    def test_native_blackwell_assembly(self):
        help_text = subprocess.run([PTXAS, '--help'], capture_output=True, text=True, check=True).stdout
        if 'sm_120' not in help_text:
            self.skipTest('native SM120 assembly requires CUDA 12.8 or newer')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'kernel.ptx'
            cubin = Path(directory) / 'kernel.cubin'
            source.write_text(self.ptx(arch='sm_120'))
            result = subprocess.run([PTXAS, '-arch=sm_120', str(source), '-o', str(cubin)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertGreater(cubin.stat().st_size, 0)

    @unittest.skipUnless(DEVICE, 'set DEUTERON_TEST_DEVICE to opt into local GPU correctness')
    def test_gpu_correctness_in_all_precision_modes(self):
        from benchmark import LocalRunner, compare_outputs

        with LocalRunner(int(DEVICE)) as runner:
            for mode in ('bf16x3', 'tf32x3', 'f32'):
                with self.subTest(mode=mode):
                    result = runner.run_kernel(self.launch, self.ptx(mode, runner.device_info['arch']), warmup=1, iters=1, samples=1)
                    compare_outputs(EXAMPLE, self.metadata, result['outputs'])


if __name__ == '__main__':
    unittest.main()
