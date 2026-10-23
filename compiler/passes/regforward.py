
from dataclasses import replace
from typing import Any

from emmy.compiler.ir.expr import BinaryExpr, Literal
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Init, Load, Write
from compiler.analysis import index_variables, kind, walk
from compiler.passes.vectorize import _uses_after

WIDTH = 4
MAX_VECTOR_ITERS = 8


def _subst_axis(expr: Any, axis: str, replacement: Any) -> Any:
    tag = kind(expr)
    if tag == 'Var':
        return replacement if expr.name == axis else expr
    if tag in ('Literal', 'Builtin') or isinstance(expr, int):
        return expr
    if tag == 'BinaryExpr':
        return replace(expr, left=_subst_axis(expr.left, axis, replacement), right=_subst_axis(expr.right, axis, replacement))
    if tag == 'TernaryExpr':
        return replace(expr, cond=_subst_axis(expr.cond, axis, replacement), if_true=_subst_axis(expr.if_true, axis, replacement), if_false=_subst_axis(expr.if_false, axis, replacement))
    if tag == 'CastExpr':
        return replace(expr, expr=_subst_axis(expr.expr, axis, replacement))
    if tag == 'FlatIndex':
        return replace(expr, index=tuple(_subst_axis(e, axis, replacement) for e in expr.index))
    if tag == 'FuncCallExpr':
        return replace(expr, args=[_subst_axis(e, axis, replacement) for e in expr.args])
    raise ValueError(f'unsubstitutable expression {expr!r}')


def _reduction_shape(loop: Any) -> Any:
    if len(loop.body) not in (2, 3):
        return None
    load, accum = (loop.body[0], loop.body[-1])
    if kind(load) != 'Load' or len(load.names) != 1 or kind(accum) != 'Accum':
        return None
    if not (accum.op.associative and accum.op.identity == 0) or accum.base is not None:
        return None
    square = None
    if len(loop.body) == 3:
        square = loop.body[1]
        if kind(square) != 'Assign' or not square.op.semiring_product:
            return None
        if square.args != (load.names[0], load.names[0]) or accum.value != square.name:
            return None
    elif accum.value != load.names[0]:
        return None
    return load, square, accum


def _no_writes_to(tile: Any, buffer: str) -> bool:
    for node in walk(tile.body):
        tag = kind(node)
        if tag == 'Write' and node.output == buffer:
            return False
        if tag == 'CpAsyncCopy' and node.smem == buffer:
            return False
    return True


def _block_barrier_between(tile: Any, first: int, second: int) -> bool:
    return any((kind(s) == 'Sync' and not s.warp for s in tile.body[first + 1:second]))


def _vector_index_last(index: Any, axis: str) -> bool:
    if not index:
        return False
    last = index[-1]
    return (
        kind(last) == 'BinaryExpr'
        and last.op == '*'
        and kind(last.left) == 'Literal'
        and last.left.value == WIDTH
        and kind(last.right) == 'Var'
        and last.right.name == axis
    )


def _match_first(loop: Any, tile: Any, threads: int) -> Any:
    if kind(loop) != 'StridedLoop' or loop.end is not None:
        return None
    if not tile.axes or kind(loop.start) != 'Var' or loop.start.name != tile.axes[-1].name:
        return None
    if tile.axes[-1].extent.value != threads:
        return None
    if kind(loop.step) != 'Literal' or loop.step.value != threads:
        return None
    extent = loop.axis.extent.value
    if extent <= 0 or extent % (threads * WIDTH):
        return None
    split = _reduction_shape(loop)
    if split is None:
        return None
    load, square, accum = split
    if load.dtype is not None and load.dtype.name != 'f32':
        return None
    if not load.index or kind(load.index[-1]) != 'Var' or load.index[-1].name != loop.axis.name:
        return None
    names = index_variables(load.index[:-1])
    if names is None or loop.axis.name in names or names - {axis.name for axis in tile.axes}:
        return None
    return load, square, accum


def _match_second(loop: Any, tile: Any, threads: int, extent: int) -> Any:
    if kind(loop) != 'StridedLoop' or loop.end is not None:
        return None
    if loop.axis.extent.value != extent // WIDTH:
        return None
    if (extent // WIDTH) % threads:
        return None
    trips = (extent // WIDTH) // threads
    if trips < 1 or trips > MAX_VECTOR_ITERS:
        return None
    for stmt in loop.body:
        tag = kind(stmt)
        if tag == 'Load':
            if len(stmt.names) != WIDTH or not _vector_index_last(stmt.index, loop.axis.name):
                return None
        elif tag == 'Write':
            if len(stmt.values) != WIDTH or stmt.atomic or stmt.swizzle != 'NONE':
                return None
            if not _vector_index_last(stmt.index, loop.axis.name):
                return None
        elif tag != 'Assign':
            return None
    row_axes = {axis.name for axis in tile.axes}
    for stmt in loop.body:
        if kind(stmt) in ('Load', 'Write'):
            names = index_variables(stmt.index[:-1])
            if names is None or loop.axis.name in names or names - row_axes:
                return None
    return trips


def _bound_names(tile: Any) -> set:
    bound = {'lane', 'warp', 'threadIdx.x', 'blockIdx.x', 'blockDim.x'}
    for axis in tile.axes:
        bound.add(axis.name)
    for node in walk(tile.body):
        tag = kind(node)
        if tag == 'Load':
            bound.update(node.names)
        elif tag in ('Assign', 'Accum', 'Init', 'Let'):
            bound.add(node.name)
        elif tag in ('WarpShuffle', 'TreeHalve'):
            bound.update(node.state)
            bound.update(node.state_b)
        elif tag in ('Loop', 'StridedLoop'):
            bound.add(node.axis.name)
        elif tag == 'Smem':
            bound.add(node.name)
        elif tag == 'Select':
            bound.add(node.name)
    return bound


def _forward_pair(tile: Any, i: int, j: int, threads: int, readonly: set, tag: int, bound: set) -> Any:
    first, second = tile.body[i], tile.body[j]
    match = _match_first(first, tile, threads)
    if match is None:
        return None
    load, square, accum = match
    buffer = load.input
    if buffer not in readonly:
        return None
    if second.start != first.start or second.step != first.step:
        return None
    trips = _match_second(second, tile, threads, first.axis.extent.value)
    if trips is None:
        return None
    x_loads = [s for s in second.body if kind(s) == 'Load' and s.input == buffer]
    if not x_loads:
        return None
    if any(s.dtype is not None and s.dtype.name != 'f32' for s in x_loads):
        return None
    pattern = x_loads[0].index
    if any(s.index != pattern for s in x_loads):
        return None
    if tuple(pattern[:-1]) != tuple(load.index[:-1]):
        return None
    if not _no_writes_to(tile, buffer) or not _block_barrier_between(tile, i, j):
        return None
    after_first = _uses_after(tile, i)
    if after_first is None:
        return None
    live = {load.names[0]} | ({square.name} if square is not None else set())
    if live & after_first:
        return None
    after_second = _uses_after(tile, j)
    if after_second is None:
        return None
    x_lanes = {lane for s in x_loads for lane in s.names}
    if x_lanes & after_second:
        return None
    for stmt in second.body:
        if kind(stmt) == 'Assign':
            if x_lanes & set(stmt.args):
                continue
        elif kind(stmt) == 'Write':
            if x_lanes & set(stmt.values):
                continue
        elif kind(stmt) == 'Load':
            if set(stmt.names) & x_lanes and stmt.input != buffer:
                return None
            continue
        else:
            return None
    step = first.step.value
    start = first.start

    def idxval(iteration: int) -> Any:
        if iteration == 0:
            return start
        return BinaryExpr(op='+', left=start, right=Literal(iteration * step, dtype='int'))

    xname = [[f'__rfwd_{tag}_{iteration}_{lane}' for lane in range(WIDTH)] for iteration in range(trips)]
    sqname = [[f'__rfwd_{tag}_sq{iteration}_{lane}' for lane in range(WIDTH)] for iteration in range(trips)]
    flat = {name for row in (*xname, *sqname) for name in row}
    freshen: dict[str, str] = {}
    for stmt in second.body:
        if kind(stmt) == 'Load' and stmt.input != buffer:
            for name in stmt.names:
                freshen[name] = f'{name}__rfu{tag}'
        elif kind(stmt) == 'Assign':
            freshen[stmt.name] = f'{stmt.name}__rfu{tag}'
    if flat & bound or set(freshen.values()) & (bound | flat):
        return None
    prefix = tuple(load.index[:-1])

    def lane_remap(iteration: int) -> dict[str, str]:
        remap = {}
        for s in x_loads:
            for lane, name in zip(s.names, xname[iteration]):
                remap[lane] = name
        for old, new in freshen.items():
            remap[old] = f'{new}_{iteration}'
        return remap

    taken = set(bound) | flat | {f'{new}_{iteration}' for new in freshen.values() for iteration in range(trips)}
    if len(taken) != len(bound) + len(flat) + len(freshen) * trips:
        return None

    first_stmts: list[Any] = [Init(name=accum.name, identity=float(accum.op.identity), dtype=accum.dtype)]
    for iteration in range(trips):
        index = (*prefix, BinaryExpr(op='*', left=Literal(WIDTH, dtype='int'), right=idxval(iteration)))
        first_stmts.append(Load(names=tuple(xname[iteration]), input=buffer, index=index, dtype=load.dtype))
        values = xname[iteration]
        if square is not None:
            for lane in range(WIDTH):
                first_stmts.append(replace(square, name=sqname[iteration][lane], args=(xname[iteration][lane],) * 2))
            values = sqname[iteration]
        for name in values:
            first_stmts.append(replace(accum, value=name))
    second_stmts: list[Any] = []
    for iteration in range(trips):
        remap = lane_remap(iteration)
        for stmt in second.body:
            if kind(stmt) == 'Load' and stmt.input == buffer:
                continue
            if kind(stmt) == 'Load':
                index = tuple(_subst_axis(e, second.axis.name, idxval(iteration)) for e in stmt.index)
                second_stmts.append(Load(names=tuple(remap[n] for n in stmt.names), input=stmt.input, index=index, dtype=stmt.dtype))
            elif kind(stmt) == 'Assign':
                second_stmts.append(replace(stmt, name=remap[stmt.name], args=tuple(remap.get(a, a) for a in stmt.args)))
            elif kind(stmt) == 'Write':
                index = tuple(_subst_axis(e, second.axis.name, idxval(iteration)) for e in stmt.index)
                second_stmts.append(Write(output=stmt.output, index=index, values=tuple(remap.get(v, v) for v in stmt.values), value_dtype=stmt.value_dtype, atomic=False, swizzle=stmt.swizzle))
    return first_stmts, second_stmts


def _forward_in_tile(tile: Any, threads: int, readonly: set, tag: int) -> Any:
    if (tile.block_threads or threads) != threads or threads <= 0:
        return None
    bound = _bound_names(tile)
    body = list(tile.body)
    for i, first in enumerate(body):
        if kind(first) != 'StridedLoop':
            continue
        for j in range(i + 1, len(body)):
            if kind(body[j]) != 'StridedLoop':
                continue
            pair = _forward_pair(tile, i, j, threads, readonly, tag, bound)
            if pair is None:
                continue
            first_stmts, second_stmts = pair
            merged: list[Any] = []
            for at, stmt in enumerate(body):
                if at == i:
                    merged.extend(first_stmts)
                elif at == j:
                    merged.extend(second_stmts)
                else:
                    merged.append(stmt)
            return replace(tile, body=Body(tuple(merged)))
    return None


def forward_reduction_loads_register(kernel: Any, launch: Any) -> Any:
    threads = launch['block'][0]
    readonly = {a['name'] for a in launch['args'] if not a.get('output')}
    changed = False
    tiles = []
    for tag, stmt in enumerate(kernel.body):
        if kind(stmt) != 'Tile':
            tiles.append(stmt)
            continue
        tile = _forward_in_tile(stmt, threads, readonly, tag)
        if tile is None:
            tiles.append(stmt)
        else:
            tiles.append(tile)
            changed = True
    if not changed:
        return None
    return replace(kernel, body=Body(tuple(tiles)))
