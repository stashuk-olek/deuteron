from __future__ import annotations
import argparse
import ctypes as ct
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')
import numpy as np
from compiler import Options, compile_kernel, compile_many
from compiler.validation import validate_launch
DTYPES: dict[str, Any] = {'f32': np.float32, 'f16': np.float16, 'f64': np.float64, 'i32': np.int32, 'i64': np.int64}
SCALARS: dict[str, Any] = {'f32': ct.c_float, 'f16': ct.c_uint16, 'f64': ct.c_double, 'i32': ct.c_int32, 'i64': ct.c_int64}
HANDLE: Any = ct.c_void_p
DEVICE_PTR: Any = ct.c_uint64

class CudaError(RuntimeError):
    pass

class Driver:

    def __init__(self) -> None:
        self.lib = ct.CDLL('libcuda.so.1')
        p = ct.POINTER
        signatures = {'cuInit': [ct.c_uint], 'cuDeviceGetCount': [p(ct.c_int)], 'cuDeviceGet': [p(ct.c_int), ct.c_int], 'cuDeviceGetName': [ct.c_char_p, ct.c_int, ct.c_int], 'cuDeviceGetAttribute': [p(ct.c_int), ct.c_int, ct.c_int], 'cuDeviceGetPCIBusId': [ct.c_char_p, ct.c_int, ct.c_int], 'cuCtxCreate_v2': [p(HANDLE), ct.c_uint, ct.c_int], 'cuCtxSetCurrent': [HANDLE], 'cuCtxDestroy_v2': [HANDLE], 'cuStreamCreate': [p(HANDLE), ct.c_uint], 'cuStreamDestroy_v2': [HANDLE], 'cuStreamSynchronize': [HANDLE], 'cuModuleLoadDataEx': [p(HANDLE), HANDLE, ct.c_uint, p(ct.c_int), p(HANDLE)], 'cuModuleGetFunction': [p(HANDLE), HANDLE, ct.c_char_p], 'cuModuleUnload': [HANDLE], 'cuFuncSetAttribute': [HANDLE, ct.c_int, ct.c_int], 'cuFuncGetAttribute': [p(ct.c_int), ct.c_int, HANDLE], 'cuMemAlloc_v2': [p(DEVICE_PTR), ct.c_size_t], 'cuMemFree_v2': [DEVICE_PTR], 'cuMemcpyHtoD_v2': [DEVICE_PTR, HANDLE, ct.c_size_t], 'cuMemcpyDtoH_v2': [HANDLE, DEVICE_PTR, ct.c_size_t], 'cuMemsetD8Async': [DEVICE_PTR, ct.c_ubyte, ct.c_size_t, HANDLE], 'cuLaunchKernel': [HANDLE, *[ct.c_uint] * 7, HANDLE, p(HANDLE), p(HANDLE)], 'cuEventCreate': [p(HANDLE), ct.c_uint], 'cuEventRecord': [HANDLE, HANDLE], 'cuEventSynchronize': [HANDLE], 'cuEventElapsedTime': [p(ct.c_float), HANDLE, HANDLE], 'cuEventDestroy_v2': [HANDLE], 'cuStreamBeginCapture_v2': [HANDLE, ct.c_int], 'cuStreamEndCapture': [HANDLE, p(HANDLE)], 'cuGraphInstantiateWithFlags': [p(HANDLE), HANDLE, ct.c_uint64], 'cuGraphLaunch': [HANDLE, HANDLE], 'cuGraphExecDestroy': [HANDLE], 'cuGraphDestroy': [HANDLE], 'cuGetErrorName': [ct.c_int, p(ct.c_char_p)], 'cuGetErrorString': [ct.c_int, p(ct.c_char_p)]}
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = ct.c_int
        self.call('cuInit', 0)

    def call(self, name: Any, *args: Any) -> None:
        status = getattr(self.lib, name)(*args)
        if status:
            error, description = (ct.c_char_p(), ct.c_char_p())
            self.lib.cuGetErrorName(status, ct.byref(error))
            self.lib.cuGetErrorString(status, ct.byref(description))
            raise CudaError(f"{name}: {(error.value or b'unknown').decode()} ({status}): {(description.value or b'').decode()}")

def argument_array(arg: Any, produced: Any = None) -> Any:
    dtype = DTYPES[arg['dtype']]
    shape = tuple(arg['shape'])
    if arg.get('intermediate'):
        if produced is None or arg['name'] not in produced:
            raise ValueError(f"missing intermediate buffer {arg['name']}")
        array = np.asarray(produced[arg['name']])
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"intermediate buffer {arg['name']} must have shape {shape} and dtype {np.dtype(dtype)}")
    elif arg.get('output'):
        array = np.zeros(shape, dtype=dtype)
    elif 'data' in arg:
        array = np.asarray(arg['data'], dtype=dtype)
    elif 'seed' in arg:
        if type(arg['seed']) is not int or arg['seed'] < 0:
            raise ValueError(f"buffer {arg['name']} requires a nonnegative integer seed")
        array = np.random.default_rng(arg['seed']).standard_normal(shape).astype(dtype)
    else:
        raise ValueError(f"buffer {arg['name']} has no seed, data, output, or intermediate")
    return np.ascontiguousarray(np.asarray(array, dtype=dtype).reshape(shape)).reshape(shape)

def launch_arguments(launch: Any) -> Any:
    return validate_launch(launch)

def validate_timing(warmup: Any, iters: Any, samples: Any, *, remote: bool = False) -> None:
    if any(type(value) is not int for value in (warmup, iters, samples)) or warmup < 0 or iters < 1 or samples < 1:
        raise ValueError('warmup must be a nonnegative integer and iters/samples must be positive integers')
    if remote and (warmup > 20 or iters > 200):
        raise ValueError('remote runner allows warmup<=20 and iters<=200')

def scalar_argument(arg: Any) -> Any:
    value = arg.get('value')
    dtype = arg['dtype']
    if type(value) not in (int, float):
        raise ValueError(f"scalar {arg['name']} requires a numeric value")
    if dtype in ('i32', 'i64'):
        bits = int(dtype[1:])
        if type(value) is not int or not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
            raise ValueError(f"scalar {arg['name']} must fit {dtype}")
    try:
        if dtype == 'f16':
            value = np.float16(value).view(np.uint16).item()
        return SCALARS[dtype](value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid scalar value for {arg['name']}") from exc

def compare_outputs(problem: Any, metadata: Any, produced: Any) -> Any:
    errors = {}
    declared_shapes = {
        arg['name']: tuple(arg['shape'])
        for launch in metadata.get('kernels', [])
        for arg in launch['args'] if arg.get('output') and not arg.get('scalar')
    }
    with np.load(Path(problem) / 'expected.npz') as expected:
        for name in expected.files:
            if name not in produced:
                raise AssertionError(f'kernel did not produce expected buffer {name}')
            actual = np.asarray(produced[name])
            reference = expected[name]
            declared = declared_shapes.get(name)
            if declared is not None and actual.shape != declared:
                raise AssertionError(f'{Path(problem).name}/{name}: output shape {actual.shape} does not match declared {declared}')
            # The original reductions serialize their one-element buffer's
            # reference as a NumPy scalar. Only normalize that explicit ABI case.
            if reference.shape == () and actual.shape == declared == (1,):
                actual = actual.reshape(())
            if actual.shape != reference.shape:
                raise AssertionError(f'{Path(problem).name}/{name}: output shape {actual.shape} does not match {reference.shape}')
            errors[name] = float(np.max(np.abs(actual.astype(np.float64) - reference))) if actual.size else 0.0
            np.testing.assert_allclose(actual, reference, rtol=metadata['rtol'], atol=metadata['atol'], equal_nan=False, err_msg=f'{Path(problem).name}/{name}')
    return errors

class LocalRunner:

    def __init__(self, device: Any = 0, module_cache_size: Any = 32) -> None:
        if type(device) is not int or device < 0:
            raise ValueError('device must be a nonnegative integer')
        if type(module_cache_size) is not int or module_cache_size < 1:
            raise ValueError('module_cache_size must be a positive integer')
        self.driver = Driver()
        self.device = device
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.context, self.stream = (HANDLE(), HANDLE())
        self.modules = OrderedDict()
        self.module_cache_size = module_cache_size
        ordinal = ct.c_int()
        self.driver.call('cuDeviceGet', ct.byref(ordinal), self.device)
        self.driver.call('cuCtxCreate_v2', ct.byref(self.context), 0, ordinal)
        try:
            self.driver.call('cuStreamCreate', ct.byref(self.stream), 1)
            name, pci = (ct.create_string_buffer(256), ct.create_string_buffer(32))
            self.driver.call('cuDeviceGetName', name, len(name), ordinal)
            self.driver.call('cuDeviceGetPCIBusId', pci, len(pci), ordinal)
            major, minor = (ct.c_int(), ct.c_int())
            self.driver.call('cuDeviceGetAttribute', ct.byref(major), 75, ordinal)
            self.driver.call('cuDeviceGetAttribute', ct.byref(minor), 76, ordinal)
            self.device_info = {'ordinal': self.device, 'name': name.value.decode(), 'pci_bus_id': pci.value.decode(), 'arch': f'sm_{major.value}{minor.value}'}
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self.context:
            return
        self._assert_owner()
        self.driver.lib.cuCtxSetCurrent(self.context)
        if self.stream:
            self.driver.lib.cuStreamSynchronize(self.stream)
        for module in self.modules.values():
            self.driver.lib.cuModuleUnload(module)
        self.modules.clear()
        if self.stream:
            self.driver.lib.cuStreamDestroy_v2(self.stream)
            self.stream = HANDLE()
        self.driver.lib.cuCtxDestroy_v2(self.context)
        self.context = HANDLE()

    def _assert_owner(self) -> None:
        if self.pid != os.getpid() or self.thread != threading.get_ident() or not self.context:
            raise RuntimeError('LocalRunner must be open and used in its creating process and thread')

    def _function(self, ptx: Any, entry: Any) -> Any:
        source = ptx.encode() if isinstance(ptx, str) else ptx
        key = hashlib.sha256(source).digest()
        if key in self.modules:
            self.modules.move_to_end(key)
            module = self.modules[key]
        else:
            module = HANDLE()
            error_log = ct.create_string_buffer(16384)
            options = (ct.c_int * 2)(5, 6)
            values = (HANDLE * 2)(ct.cast(error_log, HANDLE), HANDLE(len(error_log)))
            try:
                self.driver.call('cuModuleLoadDataEx', ct.byref(module), ct.cast(ct.c_char_p(source), HANDLE), len(options), options, values)
            except CudaError as exc:
                raise CudaError(f"{exc}\n{error_log.value.decode(errors='replace')}") from exc
            self.modules[key] = module
            if len(self.modules) > self.module_cache_size:
                _, evicted = self.modules.popitem(last=False)
                self.driver.call('cuModuleUnload', evicted)
        function = HANDLE()
        self.driver.call('cuModuleGetFunction', ct.byref(function), module, entry.encode())
        return function

    def run_kernel(self, launch: Any, ptx: Any, produced: Any = None, *, warmup: Any = 10, iters: Any = 100, samples: Any = 7) -> dict[str, Any]:
        self._assert_owner()
        validate_timing(warmup, iters, samples)
        by_name = launch_arguments(launch)
        arrays = {name: argument_array(arg, produced) for name, arg in by_name.items() if not arg.get('scalar')}
        scalars = {name: scalar_argument(arg) for name, arg in by_name.items() if arg.get('scalar')}
        self.driver.call('cuCtxSetCurrent', self.context)
        started = time.perf_counter()
        function = self._function(ptx, launch['entry'])
        jit_seconds = time.perf_counter() - started
        smem = launch.get('smem_bytes', 0)
        if smem:
            self.driver.call('cuFuncSetAttribute', function, 8, smem)
        registers, static_smem = (ct.c_int(), ct.c_int())
        self.driver.call('cuFuncGetAttribute', ct.byref(registers), 4, function)
        self.driver.call('cuFuncGetAttribute', ct.byref(static_smem), 1, function)
        allocations, host, values = ({}, {}, [])
        events, graph, executable = ([], HANDLE(), HANDLE())
        capturing = False
        try:
            for name in launch['arg_order']:
                arg = by_name[name]
                if arg.get('scalar'):
                    values.append(scalars[name])
                else:
                    array = arrays[name]
                    pointer = DEVICE_PTR()
                    self.driver.call('cuMemAlloc_v2', ct.byref(pointer), max(array.nbytes, 1))
                    allocations[name], host[name] = (pointer, array)
                    self.driver.call('cuMemcpyHtoD_v2', pointer, array.ctypes.data, array.nbytes)
                    values.append(pointer)
            params = (HANDLE * len(values))(*(ct.cast(ct.byref(value), HANDLE) for value in values))
            reset = [(allocations[name], host[name].nbytes) for name in launch.get('zero_outputs', [])]
            grid, block = (launch['grid'], launch['block'])

            def enqueue() -> None:
                for pointer, nbytes in reset:
                    self.driver.call('cuMemsetD8Async', pointer, 0, nbytes, self.stream)
                self.driver.call('cuLaunchKernel', function, *grid, *block, smem, self.stream, params, None)
            for _ in range(warmup):
                enqueue()
            self.driver.call('cuStreamSynchronize', self.stream)
            self.driver.call('cuStreamBeginCapture_v2', self.stream, 0)
            capturing = True
            for _ in range(iters):
                enqueue()
            self.driver.call('cuStreamEndCapture', self.stream, ct.byref(graph))
            capturing = False
            self.driver.call('cuGraphInstantiateWithFlags', ct.byref(executable), graph, 0)
            self.driver.call('cuGraphLaunch', executable, self.stream)
            self.driver.call('cuStreamSynchronize', self.stream)
            for _ in range(2):
                event = HANDLE()
                self.driver.call('cuEventCreate', ct.byref(event), 0)
                events.append(event)
            timings = []
            for _ in range(samples):
                self.driver.call('cuEventRecord', events[0], self.stream)
                self.driver.call('cuGraphLaunch', executable, self.stream)
                self.driver.call('cuEventRecord', events[1], self.stream)
                self.driver.call('cuEventSynchronize', events[1])
                elapsed = ct.c_float()
                self.driver.call('cuEventElapsedTime', ct.byref(elapsed), *events)
                timings.append(elapsed.value * 1000 / iters)
            outputs = {}
            for name, arg in by_name.items():
                if arg.get('output'):
                    array = np.empty_like(host[name])
                    self.driver.call('cuMemcpyDtoH_v2', array.ctypes.data, allocations[name], array.nbytes)
                    outputs[name] = array
            return {'entry': launch['entry'], 'median_us': statistics.median(timings), 'min_us': min(timings), 'samples_us': timings, 'iters': iters, 'samples': samples, 'warmup': warmup, 'timing_method': 'cuda_graph_batch_events', 'timing_includes_output_resets': bool(reset), 'registers': registers.value, 'static_smem_bytes': static_smem.value, 'dynamic_smem_bytes': smem, 'jit_seconds': jit_seconds, 'outputs': outputs}
        finally:
            if capturing:
                self.driver.lib.cuStreamEndCapture(self.stream, ct.byref(graph))
            self.driver.lib.cuStreamSynchronize(self.stream)
            for event in events:
                self.driver.lib.cuEventDestroy_v2(event)
            if executable:
                self.driver.lib.cuGraphExecDestroy(executable)
            if graph:
                self.driver.lib.cuGraphDestroy(graph)
            for pointer in allocations.values():
                self.driver.lib.cuMemFree_v2(pointer)

def write_json(path: Any, value: Any) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def reserve_artifacts(path: Path) -> None:
    if (path / 'manifest.json').exists():
        raise ValueError(f'use a new artifacts directory to preserve previous measurements: {path}')
    try:
        with (path / '.deuteron-run').open('x') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'started': time.time()}) + '\n')
    except FileExistsError as exc:
        raise ValueError(f'artifacts directory is already reserved by another run: {path}') from exc

class RemoteRunner:

    def __init__(self, server: Any, *, compile_cache_size: int = 32, arch: str | None = None) -> None:
        if type(compile_cache_size) is not int or compile_cache_size < 1:
            raise ValueError('compile_cache_size must be a positive integer')
        self.server = server.rstrip('/')
        self.device_info = {'server': self.server, 'arch': None}
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.compile_cache_size = compile_cache_size
        self.arch = arch

    def close(self) -> None:
        pass

    def request(self, endpoint: Any, body: Any) -> Any:
        command = ['curl', '--fail-with-body', '--silent', '--show-error', '--connect-timeout', '10', '--max-time', '300', '-H', 'Content-Type: application/json', '--data-binary', '@-', self.server + endpoint]
        try:
            response = subprocess.run(command, input=json.dumps(body), text=True, capture_output=True, timeout=310, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'{endpoint}: {exc}') from exc
        if response.returncode:
            raise RuntimeError(f'{endpoint}: {response.stderr or response.stdout}')
        try:
            result = json.loads(response.stdout)
        except ValueError as exc:
            raise RuntimeError(f'{endpoint}: runner returned invalid JSON') from exc
        if not isinstance(result, dict):
            raise RuntimeError(f'{endpoint}: runner response must be a JSON object')
        return result

    def _assemble(self, ptx: str, *, force: bool = False) -> Any:
        key = hashlib.sha256(ptx.encode()).hexdigest()
        if force:
            self.cache.pop(key, None)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        target = re.search(r'^\.target\s+(sm_[0-9]+a?)\b', ptx, re.M)
        if target is None:
            raise ValueError('PTX must contain a .target directive')
        arch = target.group(1) if target.group(1).endswith('a') else self.arch
        body = {'ptx': ptx}
        if arch is not None:
            body['arch'] = arch
        assembled = self.request('/compile', body)
        if assembled.get('ok') is not True or not isinstance(assembled.get('cubin_id'), str) or not assembled['cubin_id']:
            raise RuntimeError(f'assembly failed: {assembled}')
        if not isinstance(assembled.get('entry_points'), list) or any(not isinstance(name, str) for name in assembled['entry_points']):
            raise RuntimeError('assembly response has invalid entry_points')
        if not isinstance(assembled.get('registers', {}), dict):
            raise RuntimeError('assembly response has invalid registers')
        self.cache[key] = assembled
        if len(self.cache) > self.compile_cache_size:
            self.cache.popitem(last=False)
        return assembled

    def run_kernel(self, launch: Any, ptx: Any, produced: Any = None, *, warmup: Any = 20, iters: Any = 200, samples: Any = 7) -> dict[str, Any]:
        validate_timing(warmup, iters, samples, remote=True)
        by_name = launch_arguments(launch)
        arguments = []
        for name in launch['arg_order']:
            arg = dict(by_name[name])
            if arg.get('scalar'):
                scalar_argument(arg)
            elif arg.pop('intermediate', False):
                arg['data'] = argument_array(by_name[name], produced).reshape(-1).tolist()
            elif 'data' in arg and not arg.get('output'):
                argument_array(arg)
            elif 'seed' in arg and not arg.get('output'):
                if type(arg['seed']) is not int or arg['seed'] < 0:
                    raise ValueError(f'buffer {name} requires a nonnegative integer seed')
            elif not arg.get('output'):
                raise ValueError(f'buffer {name} has no seed, data, output, or intermediate')
            arguments.append(arg)
        if not isinstance(ptx, str) or not ptx.strip():
            raise ValueError('PTX must be a nonempty string')
        assembled = self._assemble(ptx)
        entry = launch['entry']
        body = {key: launch[key] for key in ('entry', 'grid', 'block')}
        body.update(cubin_id=assembled['cubin_id'], args=arguments, smem_bytes=launch.get('smem_bytes', 0), zero_outputs=launch.get('zero_outputs', []), warmup=warmup, iters=iters, max_return=max(4096, sum((math.prod(a['shape']) for a in arguments if a.get('output')))))
        recompiles = 0
        busy_responses = 0
        for attempt in range(2):
            if entry not in assembled['entry_points']:
                raise RuntimeError(f'assembly response does not contain entry {entry}')
            self.device_info['arch'] = assembled.get('arch')
            for poll in range(60):
                result = self.request('/run', body)
                if result.get('stage') != 'busy':
                    break
                busy_responses += 1
                if poll < 59:
                    time.sleep(2)
            else:
                raise RuntimeError('runner remained busy for 60 responses')
            if result.get('ok') is True:
                break
            detail = str(result.get('error', '')).lower()
            missing_cubin = re.match(r'^(?:(?:unknown|expired|evicted) cubin(?:_id| id)?\b|cubin(?:_id| id)? (?:not found|has expired|was evicted)\b)', detail) is not None
            if attempt == 0 and missing_cubin:
                assembled = self._assemble(ptx, force=True)
                body['cubin_id'] = assembled['cubin_id']
                recompiles += 1
            else:
                raise RuntimeError(f'execution failed: {result}')
        latency = result.get('median_us')
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise RuntimeError('runner returned an invalid median_us')
        expected_outputs = {name for name, arg in by_name.items() if arg.get('output')}
        if not isinstance(result.get('outputs'), dict) or set(result['outputs']) != expected_outputs:
            raise RuntimeError('runner output names disagree with the launch metadata')
        outputs = {}
        for name, output in result['outputs'].items():
            if not isinstance(output, dict):
                raise RuntimeError(f'runner returned an invalid output for {name}')
            if output.get('truncated'):
                raise RuntimeError(f'runner truncated {name}')
            shape = output.get('shape')
            if not isinstance(shape, (list, tuple)) or any(type(extent) is not int for extent in shape) or tuple(shape) != tuple(by_name[name]['shape']):
                raise RuntimeError(f'runner output shape for {name} disagrees with the launch metadata')
            try:
                outputs[name] = np.asarray(output['values'], dtype=DTYPES[by_name[name]['dtype']]).reshape(shape)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f'runner returned invalid values for {name}') from exc
        return {'entry': entry, 'median_us': latency, 'registers': assembled.get('registers', {}).get(entry), 'cubin_id': assembled['cubin_id'], 'arch': assembled.get('arch'), 'iters': iters, 'warmup': warmup, 'timing_method': 'remote_runner', 'timing_includes_output_resets': bool(launch.get('zero_outputs')), 'busy_responses': busy_responses, 'recompiles': recompiles, 'outputs': outputs}

def run_partition(device: Any, problems: Any, settings: Any) -> Any:
    runner = RemoteRunner(settings['server']) if settings['remote'] else LocalRunner(device)
    reports = []
    overrides = {k: settings[k] for k in ('contraction_precision', 'approximate_math') if settings[k] is not None}
    options = dataclasses.replace(Options(), **overrides)
    try:
        batched, batch_seconds = ({}, 0.0)
        if settings['jobs'] > 1:
            order = []
            for problem in map(Path, problems):
                for launch in json.loads((problem / 'meta.json').read_text())['kernels']:
                    order.append((problem, launch['entry']))
            arch = settings['arch'] or ('sm_89' if settings['remote'] else runner.device_info['arch'])
            started = time.perf_counter()
            results = compile_many([(problem / (entry + '.ir.json'), arch, settings['opt_level'], options) for problem, entry in order], settings['jobs'])
            batch_seconds = (time.perf_counter() - started) / len(results)
            batched = dict(zip(order, results))
        for problem in map(Path, problems):
            destination = Path(settings['artifacts']) / problem.name
            destination.mkdir(parents=True, exist_ok=True)
            produced, records = ({}, [])
            report = {'problem': problem.name, 'device': runner.device_info, 'options': dataclasses.asdict(options), 'opt_level': settings['opt_level'], 'compiler_sha256': settings['compiler_sha256']}
            try:
                metadata = json.loads((problem / 'meta.json').read_text())
                report.update(rtol=metadata['rtol'], atol=metadata['atol'])
                for launch in metadata['kernels']:
                    entry = launch['entry']
                    path = problem / (entry + '.ir.json')
                    if (problem, entry) in batched:
                        compilation, seconds = (batched[(problem, entry)], batch_seconds)
                    else:
                        started = time.perf_counter()
                        compilation = compile_kernel(path, arch=settings['arch'] or ('sm_89' if settings['remote'] else runner.device_info['arch']), opt_level=settings['opt_level'], options=options)
                        seconds = time.perf_counter() - started
                    for suffix, content in [('ptx', compilation.ptx), ('ll', compilation.optimized), ('unoptimized.ll', compilation.unoptimized)]:
                        (destination / f'{entry}.{suffix}').write_text(content)
                    record = runner.run_kernel(launch, compilation.ptx, produced, warmup=settings['warmup'], iters=settings['iters'], samples=settings['samples'])
                    produced.update(record.pop('outputs'))
                    np.savez(destination / 'actual.npz', **produced)
                    record.update(compile_seconds=seconds, ptx_sha256=hashlib.sha256(compilation.ptx.encode()).hexdigest(), ir_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    records.append(record)
                report.update(passed=True, max_abs_error=compare_outputs(problem, metadata, produced), kernels=records)
                print(f"PASS {problem.name}: {sum((r['median_us'] for r in records)):.3f} us", flush=True)
            except Exception as error:
                report.update(passed=False, error=str(error), kernels=records)
                print(f'FAIL {problem.name}: {error}', file=sys.stderr, flush=True)
                if isinstance(error, CudaError):
                    runner.close()
                    runner = LocalRunner(device)
            write_json(destination / 'result.json', report)
            reports.append(report)
    finally:
        runner.close()
    return reports

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='*')
    parser.add_argument('--problems-dir', required=True, type=Path)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--artifacts', type=Path, default=Path('benchmark-results'))
    parser.add_argument('--arch')
    parser.add_argument('--opt-level', type=int, choices=range(4), default=3)
    parser.add_argument('--contraction-precision', choices=('f32', 'tf32x3', 'bf16x3'))
    parser.add_argument('--approximate-math', action=argparse.BooleanOptionalAction)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.samples < 1:
        parser.error('warmup must be nonnegative and iters/samples positive')
    if args.jobs < 1:
        parser.error('jobs must be positive')
    if args.remote and (args.warmup > 20 or args.iters > 200):
        parser.error('remote runner allows warmup<=20 and iters<=200')
    if not args.problems_dir.is_dir():
        parser.error(f'problems directory does not exist: {args.problems_dir}')
    available = sorted((p for p in args.problems_dir.resolve().iterdir() if (p / 'meta.json').is_file()))
    problems = []
    for name in args.problems:
        matches = [p for p in available if p.name == name] or [p for p in available if p.name.startswith(name)]
        if len(matches) != 1:
            parser.error(f'problem {name!r} must match one directory')
        problems.append(matches[0])
    problems = problems or available
    if not problems or len(set(problems)) != len(problems):
        parser.error('select one or more distinct problems')
    try:
        devices = [int(d) for d in args.devices.split(',')]
    except ValueError:
        parser.error('devices must be comma-separated integers')
    if min(devices) < 0 or len(set(devices)) != len(devices):
        parser.error('devices must be distinct nonnegative ordinals')
    if args.remote:
        devices = [None]
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    try:
        reserve_artifacts(args.artifacts)
    except ValueError as exc:
        parser.error(str(exc))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source_root = Path(__file__).parent
    sources = [source_root / name for name in ('compile.py', 'benchmark.py')]
    sources.extend(sorted((source_root / 'compiler').rglob('*.py')))
    settings['compiler_sha256'] = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_json(args.artifacts / 'manifest.json', settings)
    partitions = [(d, problems[i::len(devices)], settings) for i, d in enumerate(devices) if problems[i::len(devices)]]
    if len(partitions) == 1:
        reports = run_partition(*partitions[0])
    else:
        with ProcessPoolExecutor(max_workers=len(partitions), mp_context=multiprocessing.get_context('spawn')) as pool:
            reports = [report for future in [pool.submit(run_partition, *part) for part in partitions] for report in future.result()]
    reports.sort(key=lambda report: report['problem'])
    write_json(args.artifacts / 'results.json', reports)
    print(f"{sum((r['passed'] for r in reports))}/{len(reports)} passed")
    return int(not all((r['passed'] for r in reports)))
if __name__ == '__main__':
    raise SystemExit(main())
