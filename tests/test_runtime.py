import copy
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import LocalRunner, RemoteRunner, argument_array, compare_outputs, reserve_artifacts, scalar_argument, write_json


PTX = '.version 8.7\n.target sm_120\n.address_size 64\n'
LAUNCH: dict[str, Any] = {
    'entry': 'copy', 'grid': [1, 1, 1], 'block': [32, 1, 1],
    'args': [
        {'name': 'x', 'dtype': 'f32', 'shape': [2], 'data': [1.0, 2.0]},
        {'name': 'y', 'dtype': 'f32', 'shape': [2], 'output': True},
    ],
    'arg_order': ['x', 'y'], 'zero_outputs': [], 'smem_bytes': 0,
}


def assembly(cubin='cubin-1'):
    return {'ok': True, 'cubin_id': cubin, 'entry_points': ['copy'], 'arch': 'sm_120', 'registers': {'copy': 12}}


def result():
    return {'ok': True, 'median_us': 3.0, 'outputs': {'y': {'shape': [2], 'values': [1.0, 2.0]}}}


class ScriptedRunner(RemoteRunner):
    def __init__(self, responses, **kwargs):
        super().__init__('http://unused.invalid', **kwargs)
        self.responses = iter(responses)
        self.requests = []

    def request(self, endpoint, body):
        self.requests.append((endpoint, copy.deepcopy(body)))
        expected, response = next(self.responses)
        if endpoint != expected:
            raise AssertionError(f'expected {expected}, got {endpoint}')
        return copy.deepcopy(response)


class RemoteRunnerTests(unittest.TestCase):
    def test_reuses_assembly_and_preserves_launch(self):
        runner = ScriptedRunner([('/compile', assembly()), ('/run', result()), ('/run', result())])
        launch = copy.deepcopy(LAUNCH)
        launch['zero_outputs'] = ['y']
        original = copy.deepcopy(launch)
        for _ in range(2):
            response = runner.run_kernel(launch, PTX, warmup=0, iters=1)
            np.testing.assert_array_equal(response['outputs']['y'], [1.0, 2.0])
            self.assertTrue(response['timing_includes_output_resets'])
        self.assertEqual(launch, original)
        self.assertEqual([r[0] for r in runner.requests], ['/compile', '/run', '/run'])
        for _, body in runner.requests[1:]:
            self.assertEqual(body['grid'], launch['grid'])
            self.assertEqual(body['block'], launch['block'])
            self.assertEqual(body['args'], launch['args'])
            self.assertEqual(body['zero_outputs'], ['y'])

    def test_native_architecture_suffix_is_not_rewritten(self):
        runner = ScriptedRunner([('/compile', assembly()), ('/run', result())], arch='sm_120')
        ptx = PTX.replace('sm_120', 'sm_120a')
        runner.run_kernel(LAUNCH, ptx)
        self.assertEqual(runner.requests[0][1], {'ptx': ptx, 'arch': 'sm_120a'})

    def test_python_tuple_shapes_accept_json_list_shapes(self):
        launch = copy.deepcopy(LAUNCH)
        for arg in launch['args']:
            arg['shape'] = tuple(arg['shape'])
        runner = ScriptedRunner([('/compile', assembly()), ('/run', result())])
        response = runner.run_kernel(launch, PTX)
        self.assertEqual(response['outputs']['y'].shape, (2,))

    def test_stale_cubin_recompiles_exactly_once(self):
        stale = {'ok': False, 'error': 'unknown cubin_id'}
        runner = ScriptedRunner([('/compile', assembly()), ('/run', stale), ('/compile', assembly('cubin-2')), ('/run', result())])
        response = runner.run_kernel(LAUNCH, PTX)
        self.assertEqual(response['recompiles'], 1)
        self.assertEqual(response['cubin_id'], 'cubin-2')
        self.assertEqual(runner.requests[-1][1]['cubin_id'], 'cubin-2')
        runner = ScriptedRunner([('/compile', assembly()), ('/run', stale), ('/compile', assembly('cubin-2')), ('/run', stale)])
        with self.assertRaisesRegex(RuntimeError, 'execution failed'):
            runner.run_kernel(LAUNCH, PTX)
        self.assertEqual(len(runner.requests), 4)

    def test_execution_error_is_not_retried(self):
        for error in ('CUDA illegal address', 'cubin execution failed: unknown CUDA error', 'unknown CUDA error in cubin execution'):
            with self.subTest(error=error):
                runner = ScriptedRunner([('/compile', assembly()), ('/run', {'ok': False, 'error': error})])
                with self.assertRaisesRegex(RuntimeError, 'execution failed'):
                    runner.run_kernel(LAUNCH, PTX)
                self.assertEqual(len(runner.requests), 2)

    def test_busy_retries_are_bounded_and_reported(self):
        busy = {'ok': False, 'stage': 'busy'}
        runner = ScriptedRunner([('/compile', assembly()), ('/run', busy), ('/run', result())])
        with patch('benchmark.time.sleep') as sleep:
            response = runner.run_kernel(LAUNCH, PTX)
        self.assertEqual(response['busy_responses'], 1)
        sleep.assert_called_once_with(2)
        runner = ScriptedRunner([('/compile', assembly()), *[('/run', busy)] * 60])
        with patch('benchmark.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'busy for 60'):
                runner.run_kernel(LAUNCH, PTX)
        self.assertEqual(sleep.call_count, 59)

    def test_invalid_timing_or_launch_fails_before_network(self):
        invalid = [dict(warmup=-1), dict(iters=0), dict(samples=0), dict(warmup=True), dict(iters=1.5), dict(iters=201), dict(warmup=21)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                runner = ScriptedRunner([])
                with self.assertRaises(ValueError):
                    runner.run_kernel(LAUNCH, PTX, **kwargs)
                self.assertEqual(runner.requests, [])
        launch = copy.deepcopy(LAUNCH)
        launch['arg_order'].append('x')
        runner = ScriptedRunner([])
        with self.assertRaises(ValueError):
            runner.run_kernel(launch, PTX)
        self.assertEqual(runner.requests, [])

    def test_invalid_responses_are_not_accepted_as_measurements(self):
        for replacement in ({'median_us': float('nan')}, {'median_us': -1}, {'median_us': True}, {'outputs': {}}, {'outputs': {'y': {'shape': [1, 2], 'values': [1, 2]}}}, {'outputs': {'y': {'shape': [2], 'values': [1]}}}, {'outputs': {'y': {'shape': [2], 'values': [1, 2], 'truncated': True}}}):
            with self.subTest(replacement=replacement):
                runner = ScriptedRunner([('/compile', assembly()), ('/run', {**result(), **replacement})])
                with self.assertRaises(RuntimeError):
                    runner.run_kernel(LAUNCH, PTX)

    def test_compile_cache_is_bounded(self):
        runner = ScriptedRunner([('/compile', assembly()), ('/run', result()), ('/compile', assembly('cubin-2')), ('/run', result()), ('/compile', assembly()), ('/run', result())], compile_cache_size=1)
        for ptx in (PTX, PTX + '\n', PTX):
            runner.run_kernel(LAUNCH, ptx)
            self.assertEqual(len(runner.cache), 1)
        self.assertEqual(sum(endpoint == '/compile' for endpoint, _ in runner.requests), 3)

    def test_intermediate_has_exact_shape_and_dtype(self):
        launch = copy.deepcopy(LAUNCH)
        launch['args'][0] = {'name': 'x', 'dtype': 'f32', 'shape': [2], 'intermediate': True}
        runner = ScriptedRunner([('/compile', assembly()), ('/run', result())])
        runner.run_kernel(launch, PTX, {'x': np.array([1, 2], dtype=np.float32)})
        self.assertEqual(runner.requests[1][1]['args'][0]['data'], [1.0, 2.0])
        for produced in (None, {}, {'x': np.ones((1, 2), np.float32)}, {'x': np.ones(2, np.float64)}):
            with self.subTest(produced=produced):
                runner = ScriptedRunner([])
                with self.assertRaisesRegex(ValueError, 'intermediate'):
                    runner.run_kernel(launch, PTX, produced)
                self.assertEqual(runner.requests, [])

    def test_malformed_input_fails_before_network(self):
        invalid = [
            {'name': 'x', 'dtype': 'f32', 'shape': [2]},
            {'name': 'x', 'dtype': 'f32', 'shape': [2], 'seed': None},
            {'name': 'x', 'dtype': 'f32', 'shape': [2], 'seed': -1},
            {'name': 'x', 'dtype': 'f32', 'shape': [2], 'data': [1]},
            {'name': 'x', 'dtype': 'i32', 'scalar': True},
            {'name': 'x', 'dtype': 'i32', 'scalar': True, 'value': 2**31},
        ]
        for arg in invalid:
            with self.subTest(arg=arg):
                launch = copy.deepcopy(LAUNCH)
                launch['args'][0] = arg
                runner = ScriptedRunner([])
                with self.assertRaises(ValueError):
                    runner.run_kernel(launch, PTX)
                self.assertEqual(runner.requests, [])


class LocalRuntimeSafetyTests(unittest.TestCase):
    def test_scalar_buffer_shape_is_preserved(self):
        for arg in ({'name': 'x', 'dtype': 'f32', 'shape': [], 'data': [1.]}, {'name': 'y', 'dtype': 'f32', 'shape': [], 'output': True}):
            with self.subTest(arg=arg):
                array = argument_array(arg)
                self.assertEqual(array.shape, ())
                self.assertEqual(array.nbytes, 4)
                self.assertTrue(array.flags.c_contiguous)

    def test_invalid_constructor_arguments_do_not_initialize_cuda(self):
        with patch('benchmark.Driver') as driver:
            for kwargs in (dict(device=-1), dict(device=True), dict(module_cache_size=0), dict(module_cache_size=1.5)):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    LocalRunner(**kwargs)
            driver.assert_not_called()

    def test_cross_thread_and_process_use_rejected_before_driver_calls(self):
        runner = object.__new__(LocalRunner)
        runner.pid = os.getpid()
        runner.thread = threading.get_ident()
        runner.context = ctypes.c_void_p(1)
        with patch('benchmark.os.getpid', return_value=runner.pid + 1):
            with self.assertRaisesRegex(RuntimeError, 'creating process and thread'):
                runner.close()
        with patch('benchmark.threading.get_ident', return_value=runner.thread + 1):
            with self.assertRaisesRegex(RuntimeError, 'creating process and thread'):
                runner.run_kernel(LAUNCH, PTX)

    def test_output_comparison_checks_shape_and_finite_values(self):
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder) / 'expected.npz', y=np.array([1., 2.], dtype=np.float32))
            metadata = {'rtol': 0., 'atol': 0.}
            self.assertEqual(compare_outputs(folder, metadata, {'y': np.array([1., 2.])}), {'y': 0.})
            for output in (np.array([[1., 2.]]), np.array([np.nan, 2.]), np.array([np.inf, 2.])):
                with self.subTest(output=output), self.assertRaises(AssertionError):
                    compare_outputs(folder, metadata, {'y': output})

    def test_original_scalar_reference_requires_declared_one_element_buffer(self):
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder) / 'expected.npz', y=np.array(3., dtype=np.float32))
            metadata = {'rtol': 0., 'atol': 0., 'kernels': [{'args': [
                {'name': 'y', 'dtype': 'f32', 'shape': [1], 'output': True},
            ]}]}
            self.assertEqual(compare_outputs(folder, metadata, {'y': np.array([3.])}), {'y': 0.})
            for output in (np.array(3.), np.array([[3.]]), np.array([4.])):
                with self.subTest(output=output), self.assertRaises(AssertionError):
                    compare_outputs(folder, metadata, {'y': output})
            with self.assertRaisesRegex(AssertionError, 'output shape'):
                compare_outputs(folder, {'rtol': 0., 'atol': 0.}, {'y': np.array([3.])})
            np.savez(Path(folder) / 'expected.npz', y=np.array([[3.]], dtype=np.float32))
            with self.assertRaisesRegex(AssertionError, 'output shape'):
                compare_outputs(folder, metadata, {'y': np.array([3.])})

    def test_atomic_report_failure_preserves_previous_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            write_json(path, {'complete': True})
            with self.assertRaises(TypeError):
                write_json(path, {'invalid': object()})
            self.assertEqual(json.loads(path.read_text()), {'complete': True})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_artifact_reservation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            barrier = threading.Barrier(2)

            def reserve():
                barrier.wait()
                try:
                    reserve_artifacts(path)
                    return True
                except ValueError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(reserve) for _ in range(2)]
                self.assertEqual(sum(future.result() for future in futures), 1)
            self.assertEqual(json.loads((path / '.deuteron-run').read_text())['pid'], os.getpid())
            with self.assertRaises(ValueError):
                reserve_artifacts(path)

    def test_scalar_conversion_preserves_valid_values_and_rejects_overflow(self):
        self.assertEqual(scalar_argument({'name': 'x', 'dtype': 'i32', 'value': -(2**31)}).value, -(2**31))
        self.assertEqual(scalar_argument({'name': 'x', 'dtype': 'i64', 'value': 2**63 - 1}).value, 2**63 - 1)
        self.assertEqual(scalar_argument({'name': 'x', 'dtype': 'f16', 'value': 1.5}).value, np.float16(1.5).view(np.uint16))
        for dtype, value in (('i32', 2**31), ('i64', -(2**63) - 1), ('i32', 1.5), ('f32', True), ('f16', None)):
            with self.subTest(dtype=dtype, value=value), self.assertRaises(ValueError):
                scalar_argument({'name': 'x', 'dtype': dtype, 'value': value})


if __name__ == '__main__':
    unittest.main()
