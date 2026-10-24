from typing import Any
from compiler.analysis import exprs, kind, walk


def half_storage_types(kernel: Any, launch: Any) -> Any:
    if not any(a['dtype'] == 'f16' for a in launch['args']):
        return None
    specs = {a['name']: a for a in launch['args']}
    if any(a['dtype'] not in ('f16', 'f32') or a.get('scalar') for a in specs.values()):
        raise NotImplementedError('f16 storage supports only f16/f32 buffer arguments')
    nodes = tuple(walk(kernel.body))
    controls = {'lane', 'warp'} | {a.name for s in nodes if kind(s) == 'Tile' for a in s.axes} | {s.axis.name for s in nodes if kind(s) in ('Loop', 'StridedLoop')}
    types: dict[str, str] = {}

    def index(e: Any) -> bool:
        tag = kind(e)
        if type(e) is int:
            return True
        if tag == 'Literal':
            return e.dtype == 'int'
        if tag == 'Var':
            return e.name in controls
        if tag == 'Builtin':
            return e.name.startswith(('threadIdx.', 'blockIdx.', 'blockDim.'))
        return tag == 'BinaryExpr' and e.op in ('+', '-', '*', '/', '//', '%', '&', '|', '^', '<<', '>>') and index(e.left) and index(e.right)

    for stmt in nodes:
        tag = kind(stmt)
        if tag not in ('Tile', 'Loop', 'StridedLoop', 'Load', 'Assign', 'Write'):
            raise NotImplementedError(f'f16 storage does not support {tag}')
        if any(not index(e) for e in exprs(stmt)):
            raise NotImplementedError('f16 storage requires integer control-only indices')
        names: tuple = ()
        dtype: str | None = None
        if tag == 'Load':
            if stmt.input not in specs:
                raise NotImplementedError('f16 storage loads require a global buffer')
            dtype = specs[stmt.input]['dtype']
            if stmt.dtype is not None and stmt.dtype.name != dtype:
                raise ValueError(f'Load dtype disagrees with buffer {stmt.input}')
            names = stmt.names
        elif tag == 'Assign':
            if stmt.dtype is None or stmt.dtype.name != 'f32':
                raise NotImplementedError('f16 storage arithmetic requires an explicit f32 result')
            if any(a not in types for a in stmt.args):
                raise ValueError('f16 storage arithmetic requires defined floating values')
            dtype, names = 'f32', (stmt.name,)
        elif tag == 'Write':
            if stmt.output not in specs or not specs[stmt.output].get('output') or stmt.atomic or stmt.swizzle != 'NONE':
                raise NotImplementedError('f16 storage requires ordinary global stores')
            if any(n not in types or stmt.value_dtype is not None and stmt.value_dtype.name != types[n] for n in stmt.values):
                raise ValueError('Write value dtype disagrees with its floating values')
        if dtype is None:
            continue
        for name in names:
            if name in controls or name in specs or name in types and types[name] != dtype:
                raise ValueError(f'inconsistent f16 storage value {name}')
            types[name] = dtype
    return types
