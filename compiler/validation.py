import ast
from collections.abc import Callable
import json
import math
from typing import Any
import numpy as np
from compiler.analysis import kind, walk
from compiler.options import Options
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import DataType, StructuredType
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Builtin, CastExpr, FlatIndex, FuncCallExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Smem, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Let, Load, Loop, Select, SelectBranch, StridedLoop, Write

_DTYPES = {'f16': 2, 'f32': 4, 'f64': 8, 'i32': 4, 'i64': 8}
_CONSTRUCTORS: dict[str, Callable[..., Any]] = {cls.__name__: cls for cls in (Dim, DataType, StructuredType, Axis, Window, BinaryExpr, Builtin, CastExpr, FlatIndex, FuncCallExpr, Literal, TernaryExpr, Var, CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync, Tile, TreeHalve, WarpShuffle, Accum, Assign, Body, Cond, Init, Let, Load, Loop, Select, SelectBranch, StridedLoop, Write, ElementwiseImpl)}
_CONSTRUCTORS['dtype'] = np.dtype


def validate_compile_arguments(arch: Any, opt_level: Any, options: Any) -> Options:
    if type(arch) is not str or arch not in ('sm_80', 'sm_86', 'sm_89', 'sm_90', 'sm_100', 'sm_110', 'sm_120'):
        raise ValueError(f'unsupported target {arch!r}; use sm_80, sm_86, sm_89, sm_90, sm_100, sm_110, or sm_120')
    if type(opt_level) is not int or opt_level not in (0, 1, 2, 3):
        raise ValueError('optimization level must be an integer 0, 1, 2, or 3')
    if options is None:
        options = Options()
    if not isinstance(options, Options):
        raise ValueError('options must be an Options instance or None')
    Options.__post_init__(options)
    return options


def validate_workers(workers: Any) -> None:
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError('workers must be a positive integer or None')


def _name(value: Any, label: str) -> str:
    if type(value) is not str or not value or '\x00' in value:
        raise ValueError(f'{label} must be a nonempty string without NUL bytes')
    return value


def _sequence(value: Any, label: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{label} must be a list or tuple')
    return value


def validate_launch(launch: Any) -> dict[str, Any]:
    if not isinstance(launch, dict):
        raise ValueError('launch metadata must be an object')
    _name(launch.get('entry'), 'launch entry')
    for name, limits in (('grid', (2**31 - 1, 65535, 65535)), ('block', (1024, 1024, 64))):
        dimensions = _sequence(launch.get(name), name)
        if len(dimensions) != 3 or any(type(n) is not int or not 1 <= n <= limit for n, limit in zip(dimensions, limits)):
            raise ValueError(f'{name} must contain three positive integer CUDA dimensions within {limits}')
    if math.prod(launch['block']) > 1024:
        raise ValueError('block must contain at most 1024 threads')
    smem = launch.get('smem_bytes', 0)
    if type(smem) is not int or not 0 <= smem < 2**32:
        raise ValueError('smem_bytes must be a nonnegative 32-bit integer')
    arguments = _sequence(launch.get('args'), 'args')
    order = _sequence(launch.get('arg_order'), 'arg_order')
    specs: dict[str, Any] = {}
    for arg in arguments:
        if not isinstance(arg, dict):
            raise ValueError('each argument must be an object')
        name = _name(arg.get('name'), 'argument name')
        if name in specs:
            raise ValueError('meta.json args and arg_order disagree: duplicate argument name')
        specs[name] = arg
        if type(arg.get('dtype')) is not str or arg['dtype'] not in _DTYPES:
            raise ValueError(f'unsupported argument dtype {arg.get("dtype")!r}')
        for flag in ('scalar', 'output', 'intermediate'):
            if flag in arg and type(arg[flag]) is not bool:
                raise ValueError(f'argument {flag} must be a boolean')
        if arg.get('scalar'):
            if arg.get('output') or arg.get('intermediate'):
                raise ValueError('scalar arguments cannot be output or intermediate buffers')
        else:
            shape = _sequence(arg.get('shape'), f'{name} shape')
            if any(type(extent) is not int or not 0 <= extent < 2**63 for extent in shape):
                raise ValueError('buffer shapes require nonnegative 64-bit integer extents')
            if math.prod(shape) * _DTYPES[arg['dtype']] >= 2**64:
                raise ValueError('buffer byte size does not fit a 64-bit address space')
    for name in order:
        _name(name, 'arg_order name')
    if len(order) != len(specs) or len(set(order)) != len(order) or set(order) != set(specs):
        raise ValueError('meta.json args and arg_order disagree')
    resets = _sequence(launch.get('zero_outputs', ()), 'zero_outputs')
    for name in resets:
        _name(name, 'zero_outputs name')
        if name not in specs or not specs[name].get('output') or specs[name].get('scalar'):
            raise ValueError('zero_outputs must name output buffers')
    if len(set(resets)) != len(resets):
        raise ValueError('zero_outputs cannot contain duplicate names')
    return specs


def validate_kernel_launch(kernel: Any, launch: Any) -> dict[str, Any]:
    specs = validate_launch(launch)
    if launch['entry'] != kernel.name:
        raise ValueError('launch entry and kernel name disagree')
    if tuple(launch['block'][1:]) != (1, 1) or tuple(launch['grid'][1:]) != (1, 1):
        raise NotImplementedError('only linear launches are supported')
    if kernel.serial:
        raise NotImplementedError('serial launch axes are outside the supported subset')
    for arg in specs.values():
        if arg['dtype'] not in ('f16', 'f32') or arg.get('scalar'):
            raise NotImplementedError('the backend accepts f16/f32 buffer arguments')
        if math.prod(arg['shape']) > 2**31:
            raise NotImplementedError('buffer element count exceeds signed 32-bit indexing')
    shared = {}
    nodes = tuple(walk(kernel.body))
    for stmt in nodes:
        if kind(stmt) == 'Smem':
            _name(stmt.name, 'shared buffer name')
            if stmt.name in specs or stmt.name in shared:
                raise ValueError(f'duplicate buffer name {stmt.name}')
            _sequence(stmt.extents, 'shared extents')
            if any(type(n) is not int or n < 0 for n in stmt.extents) or math.prod(stmt.extents) > 2**31:
                raise ValueError('shared extents must fit signed 32-bit indexing')
            if type(stmt.align) is not int or stmt.align < 0 or stmt.align and stmt.align & (stmt.align - 1):
                raise ValueError('shared alignment must be zero or a power of two')
            shared[stmt.name] = stmt
    for stmt in nodes:
        tag = kind(stmt)
        if tag in ('Tile', 'Loop', 'StridedLoop'):
            axes = stmt.axes if tag == 'Tile' else (stmt.axis,)
            for axis in axes:
                _name(axis.name, 'axis name')
                try:
                    extent = axis.extent.value
                except (AttributeError, TypeError, ValueError) as exc:
                    raise NotImplementedError('axis extents must be concrete integers') from exc
                if type(extent) is not int or not 0 <= extent < 2**31:
                    raise NotImplementedError('axis extents must be nonnegative signed 32-bit integers')
            if tag == 'Tile':
                if not axes or any(a.extent.value == 0 for a in axes) or len({a.name for a in axes}) != len(axes):
                    raise ValueError('Tile requires distinct axes with positive extents')
                if stmt.block_threads is not None and (type(stmt.block_threads) is not int or stmt.block_threads != launch['block'][0]):
                    raise ValueError('Tile block_threads disagrees with the launch')
            if tag == 'StridedLoop' and kind(stmt.step) == 'Literal' and (type(stmt.step.value) is not int or stmt.step.value <= 0):
                raise ValueError('StridedLoop requires a positive integer step')
        if tag in ('Load', 'Write', 'CpAsyncCopy'):
            accesses = ((stmt.input, stmt.index, False),) if tag == 'Load' else ((stmt.output, stmt.index, True),) if tag == 'Write' else ((stmt.src, stmt.src_index, False), (stmt.smem, stmt.smem_index, True))
            for name, index, writing in accesses:
                if name not in specs and name not in shared:
                    raise ValueError(f'unknown buffer {name}')
                shape = specs[name]['shape'] if name in specs else shared[name].extents
                if len(index) != len(shape) and not (not index and math.prod(shape) == 1):
                    raise ValueError(f'{name}: index rank does not match its shape')
                if writing and name in specs and not specs[name].get('output'):
                    raise NotImplementedError(f'ordinary global stores require output buffers: {name}')
            if tag == 'CpAsyncCopy':
                if stmt.src not in specs or stmt.smem not in shared:
                    raise ValueError('cp.async requires global source and shared destination')
                if type(stmt.nbytes) is not int or stmt.nbytes not in (4, 8, 16):
                    raise ValueError('cp.async copy size must be 4, 8, or 16')
        if tag == 'CpAsyncWait' and (type(stmt.group) is not int or not 0 <= stmt.group <= 7):
            raise ValueError('cp.async wait group must be an integer from 0 to 7')
        if tag == 'Sync':
            if type(stmt.barrier_id) is not int or not 0 <= stmt.barrier_id <= 15 or type(stmt.warp) is not bool:
                raise ValueError('barrier id must be 0..15 and warp must be boolean')
            if stmt.warp and stmt.barrier_id:
                raise ValueError('warp synchronization cannot use a named barrier')
            if stmt.count is not None and (type(stmt.count) is not int or not 0 < stmt.count <= math.prod(launch['block']) or stmt.count % 32):
                raise ValueError('barrier count must be a positive multiple of 32 within the block')
            if stmt.barrier_id and stmt.count is None:
                raise ValueError('named barrier requires a thread count')
    return specs


def _parse_repr(text: Any) -> Any:
    if type(text) is not str:
        raise ValueError('IR statements must be constructor strings')

    def parse(node: Any) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [parse(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
            return {parse(key): parse(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id in ('inf', 'nan'):
            return math.inf if node.id == 'inf' else math.nan
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = parse(node.operand)
            if type(value) in (int, float):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CONSTRUCTORS:
            if any(keyword.arg is None for keyword in node.keywords) or len({keyword.arg for keyword in node.keywords}) != len(node.keywords):
                raise ValueError('IR constructors require distinct explicit keyword arguments')
            args = [parse(arg) for arg in node.args]
            kwargs = {keyword.arg: parse(keyword.value) for keyword in node.keywords if keyword.arg is not None}
            return _CONSTRUCTORS[node.func.id](*args, **kwargs)
        raise ValueError('IR accepts only known constructors, literals, and literal containers')

    try:
        return parse(ast.parse(text, mode='eval').body)
    except (SyntaxError, TypeError, AttributeError, KeyError, IndexError, RecursionError, OverflowError) as exc:
        raise ValueError(f'invalid IR constructor expression: {exc}') from exc


def _json(payload: Any, label: str) -> Any:
    def unique(pairs: Any) -> Any:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result
    try:
        result = json.loads(payload, object_pairs_hook=unique)
        if not isinstance(result, dict):
            raise ValueError('expected an object')
        return result
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f'invalid {label}: {exc}') from exc


def parse_kernel(payload: Any, meta_payload: Any) -> tuple[Any, Any]:
    data, metadata = _json(payload, 'IR JSON'), _json(meta_payload, 'metadata JSON')
    try:
        if not isinstance(data['nodes'], dict) or any(not isinstance(node, dict) for node in data['nodes'].values()):
            raise ValueError('graph nodes must be objects')
        kernels = [node for node in data['nodes'].values() if node.get('op') == 'KernelOp']
        if len(kernels) != 1:
            raise ValueError('expected exactly one KernelOp in the input graph')
        fields = kernels[0].get('op_fields', {})
        if not isinstance(fields, dict) or set(fields) - {'body', 'name', 'zero_delegated', 'serial'}:
            raise ValueError('unsupported KernelOp fields')
        _name(fields.get('name'), 'kernel name')
        body = fields.get('body', ())
        _sequence(body, 'KernelOp body')
        serial = fields.get('serial', ())
        _sequence(serial, 'KernelOp serial')
        delegated = fields.get('zero_delegated', ())
        _sequence(delegated, 'KernelOp zero_delegated')
        for name in delegated:
            _name(name, 'zero_delegated name')
        kernel = KernelOp(body=Body(tuple(_parse_repr(item) for item in body)), name=fields['name'], zero_delegated=tuple(delegated), serial=tuple(_parse_repr(item) for item in serial))
        launches = metadata['kernels']
        _sequence(launches, 'metadata kernels')
        if any(not isinstance(launch, dict) for launch in launches):
            raise ValueError('metadata kernels must contain objects')
        matches = [launch for launch in launches if launch.get('entry') == kernel.name]
        if len(matches) != 1:
            raise ValueError(f'meta.json must contain exactly one launch for {kernel.name}')
        launch = matches[0]
        validate_kernel_launch(kernel, launch)
        return kernel, launch
    except (KeyError, TypeError, AttributeError, IndexError, RecursionError, OverflowError) as exc:
        raise ValueError(f'invalid scheduled Kernel IR or launch metadata: {exc}') from exc
