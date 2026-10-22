from __future__ import annotations
import argparse
import ctypes as ct
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')
import numpy as np
from compiler import Options, compile_kernel, compile_many
DTYPES = {'f32': np.float32, 'f16': np.float16, 'f64': np.float64, 'i32': np.int32, 'i64': np.int64}
SCALARS = {'f32': ct.c_float, 'f16': ct.c_uint16, 'f64': ct.c_double, 'i32': ct.c_int32, 'i64': ct.c_int64}
HANDLE = ct.c_void_p
DEVICE_PTR = ct.c_uint64

class CudaError(RuntimeError):
    pass

class Driver:

    def __init__(self):
        self.lib = ct.CDLL('libcuda.so.1')
        p = ct.POINTER
        signatures = {'cuInit': [ct.c_uint], 'cuDeviceGetCount': [p(ct.c_int)], 'cuDeviceGet': [p(ct.c_int), ct.c_int], 'cuDeviceGetName': [ct.c_char_p, ct.c_int, ct.c_int], 'cuDeviceGetAttribute': [p(ct.c_int), ct.c_int, ct.c_int], 'cuDeviceGetPCIBusId': [ct.c_char_p, ct.c_int, ct.c_int], 'cuCtxCreate_v2': [p(HANDLE), ct.c_uint, ct.c_int], 'cuCtxSetCurrent': [HANDLE], 'cuCtxDestroy_v2': [HANDLE], 'cuStreamCreate': [p(HANDLE), ct.c_uint], 'cuStreamDestroy_v2': [HANDLE], 'cuStreamSynchronize': [HANDLE], 'cuModuleLoadDataEx': [p(HANDLE), HANDLE, ct.c_uint, p(ct.c_int), p(HANDLE)], 'cuModuleGetFunction': [p(HANDLE), HANDLE, ct.c_char_p], 'cuModuleUnload': [HANDLE], 'cuFuncSetAttribute': [HANDLE, ct.c_int, ct.c_int], 'cuFuncGetAttribute': [p(ct.c_int), ct.c_int, HANDLE], 'cuMemAlloc_v2': [p(DEVICE_PTR), ct.c_size_t], 'cuMemFree_v2': [DEVICE_PTR], 'cuMemcpyHtoD_v2': [DEVICE_PTR, HANDLE, ct.c_size_t], 'cuMemcpyDtoH_v2': [HANDLE, DEVICE_PTR, ct.c_size_t], 'cuMemsetD8Async': [DEVICE_PTR, ct.c_ubyte, ct.c_size_t, HANDLE], 'cuLaunchKernel': [HANDLE, *[ct.c_uint] * 7, HANDLE, p(HANDLE), p(HANDLE)], 'cuEventCreate': [p(HANDLE), ct.c_uint], 'cuEventRecord': [HANDLE, HANDLE], 'cuEventSynchronize': [HANDLE], 'cuEventElapsedTime': [p(ct.c_float), HANDLE, HANDLE], 'cuEventDestroy_v2': [HANDLE], 'cuStreamBeginCapture_v2': [HANDLE, ct.c_int], 'cuStreamEndCapture': [HANDLE, p(HANDLE)], 'cuGraphInstantiateWithFlags': [p(HANDLE), HANDLE, ct.c_uint64], 'cuGraphLaunch': [HANDLE, HANDLE], 'cuGraphExecDestroy': [HANDLE], 'cuGraphDestroy': [HANDLE], 'cuGetErrorName': [ct.c_int, p(ct.c_char_p)], 'cuGetErrorString': [ct.c_int, p(ct.c_char_p)]}
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = ct.c_int
        self.call('cuInit', 0)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            error, description = (ct.c_char_p(), ct.c_char_p())
            self.lib.cuGetErrorName(status, ct.byref(error))
            self.lib.cuGetErrorString(status, ct.byref(description))
            raise CudaError(f"{name}: {(error.value or b'unknown').decode()} ({status}): {(description.value or b'').decode()}")

def argument_array(arg, produced=None):
    dtype = DTYPES[arg['dtype']]
    shape = tuple(arg['shape'])
    if arg.get('intermediate'):
        if produced is None or arg['name'] not in produced:
            raise ValueError(f"missing intermediate buffer {arg['name']}")
        array = produced[arg['name']]
    elif arg.get('output'):
        array = np.zeros(shape, dtype=dtype)
    elif 'data' in arg:
        array = np.asarray(arg['data'], dtype=dtype)
    elif 'seed' in arg:
        array = np.random.default_rng(arg['seed']).standard_normal(shape).astype(dtype)
    else:
        raise ValueError(f"buffer {arg['name']} has no seed, data, output, or intermediate")
    return np.ascontiguousarray(np.asarray(array, dtype=dtype).reshape(shape))

def launch_arguments(launch):
    by_name = {arg['name']: arg for arg in launch['args']}
    if len(by_name) != len(launch['args']) or len(by_name) != len(launch['arg_order']) or set(by_name) != set(launch['arg_order']):
        raise ValueError('launch args and arg_order must name each argument exactly once')
    return by_name

def compare_outputs(problem, metadata, produced):
    errors = {}
    with np.load(Path(problem) / 'expected.npz') as expected:
        for name in expected.files:
            if name not in produced:
                raise AssertionError(f'kernel did not produce expected buffer {name}')
            actual = produced[name].reshape(expected[name].shape)
            reference = expected[name]
            errors[name] = float(np.max(np.abs(actual.astype(np.float64) - reference))) if actual.size else 0.0
            np.testing.assert_allclose(actual, reference, rtol=metadata['rtol'], atol=metadata['atol'], equal_nan=False, err_msg=f'{Path(problem).name}/{name}')
    return errors

class LocalRunner:

    def __init__(self, device=0, module_cache_size=32):
        self.driver = Driver()
        self.device = int(device)
        self.pid = os.getpid()
        self.context, self.stream = (HANDLE(), HANDLE())
        self.modules = OrderedDict()
        self.module_cache_size = max(1, module_cache_size)
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.context:
            return
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

    def _function(self, ptx, entry):
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

    def run_kernel(self, launch, ptx, produced=None, *, warmup=10, iters=100, samples=7):
        if self.pid != os.getpid() or not self.context:
            raise RuntimeError('LocalRunner must be open and used in its creating process')
        if warmup < 0 or iters < 1 or samples < 1:
            raise ValueError('warmup must be nonnegative and iters/samples must be positive')
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
            by_name = launch_arguments(launch)
            for name in launch['arg_order']:
                arg = by_name[name]
                if arg.get('scalar'):
                    scalar = arg['value']
                    if arg['dtype'] == 'f16':
                        scalar = np.float16(scalar).view(np.uint16).item()
                    values.append(SCALARS[arg['dtype']](scalar))
                else:
                    array = argument_array(arg, produced)
                    pointer = DEVICE_PTR()
                    self.driver.call('cuMemAlloc_v2', ct.byref(pointer), max(array.nbytes, 1))
                    allocations[name], host[name] = (pointer, array)
                    self.driver.call('cuMemcpyHtoD_v2', pointer, array.ctypes.data, array.nbytes)
                    values.append(pointer)
            params = (HANDLE * len(values))(*(ct.cast(ct.byref(value), HANDLE) for value in values))
            reset = [(allocations[name], host[name].nbytes) for name in launch.get('zero_outputs', [])]
            grid, block = (launch['grid'], launch['block'])
            if len(grid) != 3 or len(block) != 3 or min(*grid, *block) < 1:
                raise ValueError('grid and block must each contain three positive dimensions')

            def enqueue():
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

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

class RemoteRunner:

    def __init__(self, server):
        self.server = server.rstrip('/')
        self.device_info = {'server': self.server, 'arch': None}

    def close(self):
        pass

    def request(self, endpoint, body):
        command = ['curl', '--fail-with-body', '--silent', '--show-error', '--connect-timeout', '10', '--max-time', '300', '-H', 'Content-Type: application/json', '--data-binary', '@-', self.server + endpoint]
        response = subprocess.run(command, input=json.dumps(body), text=True, capture_output=True, timeout=310, check=False)
        if response.returncode:
            raise RuntimeError(response.stderr or response.stdout)
        return json.loads(response.stdout)

    def run_kernel(self, launch, ptx, produced=None, *, warmup=20, iters=200, samples=7):
        by_name = launch_arguments(launch)
        assembled = self.request('/compile', {'ptx': ptx})
        entry = launch['entry']
        if not assembled.get('ok') or entry not in assembled.get('entry_points', []):
            raise RuntimeError(f'assembly failed: {assembled}')
        self.device_info['arch'] = assembled.get('arch')
        arguments = []
        for name in launch['arg_order']:
            arg = dict(by_name[name])
            if arg.pop('intermediate', False):
                arg['data'] = produced[name].reshape(-1).tolist()
            arguments.append(arg)
        body = {key: launch[key] for key in ('entry', 'grid', 'block')}
        body.update(cubin_id=assembled['cubin_id'], args=arguments, smem_bytes=launch.get('smem_bytes', 0), zero_outputs=launch.get('zero_outputs', []), warmup=warmup, iters=iters, max_return=max(4096, sum((math.prod(a['shape']) for a in arguments if a.get('output')))))
        for _ in range(60):
            result = self.request('/run', body)
            if result.get('stage') != 'busy':
                break
            time.sleep(2)
        if not result.get('ok'):
            raise RuntimeError(f'execution failed: {result}')
        outputs = {}
        for name, output in result['outputs'].items():
            if output.get('truncated'):
                raise RuntimeError(f'runner truncated {name}')
            outputs[name] = np.asarray(output['values'], dtype=DTYPES[by_name[name]['dtype']]).reshape(output['shape'])
        return {'entry': entry, 'median_us': result['median_us'], 'registers': assembled.get('registers', {}).get(entry), 'cubin_id': assembled['cubin_id'], 'arch': assembled.get('arch'), 'iters': iters, 'warmup': warmup, 'timing_method': 'remote_runner', 'timing_includes_output_resets': bool(launch.get('zero_outputs')), 'outputs': outputs}

def run_partition(device, problems, settings):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problems', nargs='*')
    parser.add_argument('--problems-dir', required=True, type=Path)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--artifacts', type=Path, default=Path('benchmark-results'))
    parser.add_argument('--arch')
    parser.add_argument('--opt-level', type=int, choices=range(4), default=2)
    parser.add_argument('--contraction-precision', choices=('f32', 'tf32x3', 'bf16x3'))
    parser.add_argument('--approximate-math', action=argparse.BooleanOptionalAction)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.samples < 1:
        parser.error('warmup must be nonnegative and iters/samples positive')
    if args.remote and (args.warmup > 20 or args.iters > 200):
        parser.error('remote runner allows warmup<=20 and iters<=200')
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
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    names = ('compile.py', 'benchmark.py', 'compiler/__init__.py', 'compiler/options.py', 'compiler/analysis.py', 'compiler/transforms.py', 'compiler/build.py', 'compiler/passes/__init__.py', 'compiler/passes/contraction.py', 'compiler/passes/recurrence.py', 'compiler/codegen/__init__.py', 'compiler/codegen/lower.py', 'compiler/codegen/tensor.py')
    settings['compiler_sha256'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in names}
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
