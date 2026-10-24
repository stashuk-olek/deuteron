import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from compiler import compile_kernel

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples' / 'vector_add'


class CompilerInterfaceTests(unittest.TestCase):
    def test_ir_and_launch_metadata_are_sufficient(self):
        original = compile_kernel(EXAMPLE / 'vector_add.ir.json', opt_level=3)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            for name in ('vector_add.ir.json', 'meta.json'):
                shutil.copy2(EXAMPLE / name, destination / name)
            isolated = compile_kernel(destination / 'vector_add.ir.json', opt_level=3)
        self.assertEqual(original.ptx, isolated.ptx)

    def test_supported_targets_compile_without_a_gpu(self):
        for arch in ('sm_80', 'sm_86', 'sm_89', 'sm_90', 'sm_100', 'sm_110', 'sm_120'):
            with self.subTest(arch=arch):
                result = compile_kernel(EXAMPLE / 'vector_add.ir.json', arch=arch, opt_level=3)
                self.assertIn('.target ' + arch, result.ptx)
                if arch == 'sm_110':
                    self.assertIn('.version 9.0', result.ptx)

    def test_duplicate_argument_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copy2(EXAMPLE / 'vector_add.ir.json', destination / 'vector_add.ir.json')
            metadata = json.loads((EXAMPLE / 'meta.json').read_text())
            metadata['kernels'][0]['arg_order'] = ['lhs', 'lhs', 'sum']
            (destination / 'meta.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'args and arg_order disagree'):
                compile_kernel(destination / 'vector_add.ir.json')

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported target'):
            compile_kernel(EXAMPLE / 'vector_add.ir.json', arch='sm_999')


@unittest.skipUnless(os.environ.get('DEUTERON_TEST_DEVICE'), 'set DEUTERON_TEST_DEVICE to a free GPU ordinal')
class VectorExecutionTests(unittest.TestCase):
    def test_partial_last_block_matches_numpy_exactly(self):
        from benchmark import LocalRunner, compare_outputs
        metadata = json.loads((EXAMPLE / 'meta.json').read_text())
        with LocalRunner(int(os.environ['DEUTERON_TEST_DEVICE'])) as runner:
            compiled = compile_kernel(EXAMPLE / 'vector_add.ir.json', arch=runner.device_info['arch'], opt_level=3)
            result = runner.run_kernel(metadata['kernels'][0], compiled.ptx, warmup=0, iters=1, samples=1)
        compare_outputs(EXAMPLE, metadata, result['outputs'])


if __name__ == '__main__':
    unittest.main()
