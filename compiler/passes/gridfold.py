import math
from dataclasses import replace
from typing import Any
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Load, Write
from compiler.analysis import STATIC_SHARED_LIMIT, canonical, difference, index_variables, kind, known_multiple, substitute, walk
from compiler.passes.contraction import _tensorize_tile

def _affine(poly: Any) -> Any:
    result = Literal(0, dtype='int')
    for term, coefficient in poly.items():
        if term == ():
            result = result + coefficient
        elif len(term) == 2 and term[0] == 'var':
            result = result + Var(term[1]) * coefficient
        else:
            raise ValueError('non-affine source origin')
    return result

def fold_grid_contractions(kernel: Any, launch: Any) -> Any:
    try:
        return _fold(kernel, launch)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return kernel

def _fold(kernel: Any, launch: Any) -> Any:
    if kernel.serial or len(kernel.body) != 1 or kind(kernel.body[0]) != 'Tile':
        return kernel
    tile = kernel.body[0]
    if tile.raster_group or tile.aux_threads or launch['grid'][1:] != [1, 1] or launch['block'][1:] != [1, 1]:
        return kernel
    threads = launch['block'][0]
    counters = tuple(tile.axes) + tuple(s.axis for s in walk(tile.body) if kind(s) in ('Loop', 'StridedLoop'))
    names = {a.name for a in counters}
    if threads <= 0 or threads % 32 or launch['grid'][0] * threads > 2**32 or any(a.extent.value <= 0 for a in counters) or len(names) != len(counters) or names & {'lane', 'warp'}:
        return kernel
    definitions = {s.name for s in walk(tile.body) if kind(s) in ('Init', 'Assign', 'Accum', 'Let')} | {n for s in walk(tile.body) if kind(s) == 'Load' for n in s.names}
    if definitions & (names | {'lane', 'warp'}) or any(n.startswith(('threadIdx.', 'blockIdx.', 'blockDim.')) for n in names | definitions):
        return kernel
    plans = _tensorize_tile(tile, launch, threads, 0, set())
    if len(plans) != 1:
        return kernel
    p = plans[0]
    if any(s.op.name != 'add' for s in p.loop.body if kind(s) == 'Accum'):
        return kernel
    outer = p.outer
    smem = {s.name: s for s in tile.body if kind(s) == 'Smem'}
    accums = {s.name for s in p.loop.body if kind(s) == 'Accum'}
    kept = {id(outer)} | {id(s) for s in p.writes} | {id(s) for s in smem.values()} | {id(s) for s in tile.body if kind(s) == 'Init' and s.name in accums}
    if set(smem) != {p.lhs_buffer, p.rhs_buffer} or any(id(s) not in kept for s in tile.body) or not p.atomic:
        return kernel
    if any(canonical(x) for x in (*p.lhs_base, *p.rhs_base)):
        return kernel
    if outer.axis.extent.value != p.depth or outer.end is not None or canonical(outer.start) or canonical(outer.step) != {(): p.depth}:
        return kernel
    if tuple(kind(s) for s in outer.body) != ('StridedLoop', 'StridedLoop', 'CpAsyncCommit', 'CpAsyncWait', 'Sync', 'Loop', 'Sync') or outer.body[3].group != 0:
        return kernel
    output_names = index_variables(p.output_base)
    if output_names is None:
        return kernel
    candidates = [a for a in tile.axes[:-2] if a.extent.value > 1 and a.name not in output_names]
    if len(candidates) != 1:
        return kernel
    split = candidates[0]
    args = {a['name']: a for a in launch['args']}
    roles = {p.lhs_buffer: 1, p.rhs_buffer: 0}
    lengths = {p.lhs_buffer: p.rows, p.rhs_buffer: p.cols}
    sources = {}
    thread_index = Var(tile.axes[-2].name) * tile.axes[-1].extent.value + Var(tile.axes[-1].name)
    for loop in outer.body[:2]:
        if len(loop.body) != 1 or kind(loop.body[0]) != 'CpAsyncCopy' or loop.end is not None or canonical(loop.start) != canonical(thread_index) or canonical(loop.step) != {(): threads}:
            return kernel
        copy = loop.body[0]
        if copy.smem not in roles or copy.smem in sources or copy.nbytes != 16 or copy.swizzle != 'NONE' or copy.lane_index is not None or copy.lane_rows:
            return kernel
        shape = smem[copy.smem].extents
        role = roles[copy.smem]
        if shape[role] != p.depth or shape[1 - role] != lengths[copy.smem] or loop.axis.extent.value * 4 != math.prod(shape):
            return kernel
        indices = (Var(loop.axis.name) * 4 / shape[1], Var(loop.axis.name) * 4 % shape[1])
        if tuple(map(canonical, copy.smem_index)) != tuple(map(canonical, indices)) or len(copy.src_index) != 2:
            return kernel
        origins = [difference(canonical(a), canonical(b)) for a, b in zip(copy.src_index, indices)]
        if origins[role] != {('var', split.name): p.depth, ('var', outer.axis.name): 1} or origins[1 - role] != canonical(p.output_base[1 - role]):
            return kernel
        source = args.get(copy.src)
        output = args[p.output_buffer]
        if source is None or source.get('output') or source['dtype'] != 'f32' or len(source['shape']) != 2 or source['shape'][role] != split.extent.value * p.depth or source['shape'][1 - role] != output['shape'][1 - role] or source['shape'][-1] % 4 or known_multiple(substitute(copy.src_index[-1], {outer.axis.name})) % 4:
            return kernel
        sources[copy.smem] = (loop, copy, origins)
    if len(sources) != 2:
        return kernel
    size = sum(math.prod(s.extents) * 4 for s in smem.values())
    factor = 1
    while split.extent.value % (factor * 2) == 0 and p.depth * factor * 2 <= 32 and launch['grid'][0] * threads // (factor * 2) >= 32768 and size * factor * 2 <= STATIC_SHARED_LIMIT:
        factor *= 2
    if factor == 1:
        return kernel
    body = []
    prefix = '__stage'
    while any(n.startswith(prefix) for n in names | definitions | set(args)):
        prefix += '_'
    for i, (loop, copy, origins) in enumerate(sources.values()):
        shape = list(smem[copy.smem].extents)
        role = roles[copy.smem]
        shape[role] *= factor
        origins[role][('var', split.name)] *= factor
        indices = (Var(loop.axis.name) * 4 / shape[1], Var(loop.axis.name) * 4 % shape[1])
        src = tuple(_affine(origin) + index for origin, index in zip(origins, indices))
        values = tuple(f'{prefix}_{i}_{j}' for j in range(4))
        copies = (Load(names=values, input=copy.src, index=src, dtype=F32), Write(output=copy.smem, index=indices, values=values, value_dtype=F32))
        body.append(replace(loop, axis=replace(loop.axis, extent=Dim(math.prod(shape) // 4)), body=Body(copies)))
    depth = p.depth * factor
    body.extend((outer.body[4], replace(p.loop, axis=replace(p.loop.axis, extent=Dim(depth)))))
    expanded = replace(outer, axis=replace(outer.axis, extent=Dim(depth)), step=Literal(depth, dtype='int'), body=Body(body))
    body = []
    for s in tile.body:
        if s is outer:
            s = expanded
        elif kind(s) == 'Smem':
            shape = list(s.extents)
            shape[roles[s.name]] *= factor
            s = replace(s, extents=tuple(shape))
        body.append(s)
    axes = tuple(replace(a, extent=Dim(a.extent.value // factor)) if a is split else a for a in tile.axes)
    return replace(kernel, body=Body((replace(tile, axes=axes, body=Body(body)),)))
