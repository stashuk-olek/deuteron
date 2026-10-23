from dataclasses import replace
from emmy.compiler.ir.axis import Axis, Dim
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.blocks import StridedLoop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Load, Write
from compiler.analysis import exprs, free_variables, index_variables, kind, uses, walk
from compiler.transforms import SCALAR_OPS


def _uses_after(tile, at):
    used = set()
    for s in tile.body[at + 1:]:
        for node in walk((s,)):
            names = uses(node)
            if names is None:
                return None
            used.update(names)
            expressions = exprs(node)
            if expressions is None:
                return None
            for expression in expressions:
                names = free_variables(expression)
                if names is None:
                    return None
                used.update(names)
    return used


def _match(loop, tile, threads, tag, at, used_after):
    if kind(loop) != 'StridedLoop':
        return None
    if not tile.axes:
        return None
    extent = loop.axis.extent.value
    if extent <= 0 or type(loop.start).__name__ != 'Var' or loop.start.name != tile.axes[-1].name:
        return None
    if type(loop.step).__name__ != 'Literal' or loop.step.value != threads or tile.axes[-1].extent.value != threads:
        return None
    if loop.end is not None:
        return None
    width = 0
    for candidate in (4,):
        if extent % (threads * candidate) == 0:
            width = candidate
            break
    if not width:
        return None
    lane_of = {}
    for s in loop.body:
        tag2 = kind(s)
        if tag2 == 'Load':
            if len(s.names) != 1:
                return None
        elif tag2 == 'Assign':
            if s.op.name not in SCALAR_OPS:
                return None
        elif tag2 == 'Write':
            if len(s.values) != 1 or s.atomic or s.swizzle != 'NONE':
                return None
        else:
            return None
    for s in loop.body:
        if kind(s) in ('Load', 'Write'):
            index = s.index
            if not index or type(index[-1]).__name__ != 'Var' or index[-1].name != loop.axis.name:
                return None
            names = index_variables(index[:-1])
            if names is None:
                return None
            if loop.axis.name in names:
                return None
            row_axes = {axis.name for axis in tile.axes}
            if names - row_axes:
                return None
    by_buffer = {}
    for s in loop.body:
        if kind(s) == 'Load':
            by_buffer.setdefault(('load', s.input), []).append(s.index)
        if kind(s) == 'Write':
            by_buffer.setdefault(('store', s.output), []).append(s.index)
    buffers = {}
    for (access, name), indices in by_buffer.items():
        buffers.setdefault(name, {})[access] = indices
    for name, access in buffers.items():
        if 'load' in access and 'store' in access:
            for left in access['load']:
                for right in access['store']:
                    if len(left) != len(right) or any((a != b for a, b in zip(left, right))):
                        return None
    defined = set()
    for s in loop.body:
        if kind(s) == 'Load':
            defined.update(s.names)
        elif kind(s) == 'Assign':
            defined.add(s.name)
    if used_after is None or (defined & used_after):
        return None
    bound = {axis.name for axis in tile.axes} | {'lane', 'warp'}
    for node in walk(tile.body):
        tag3 = kind(node)
        if tag3 == 'Load':
            bound.update(node.names)
        elif tag3 in ('Assign', 'Accum', 'Init', 'Let'):
            bound.add(node.name)
        elif tag3 in ('WarpShuffle', 'TreeHalve'):
            bound.update(node.state)
            bound.update(node.state_b)
        elif tag3 in ('Loop', 'StridedLoop'):
            bound.add(node.axis.name)
        elif tag3 == 'Smem':
            bound.add(node.name)
        elif tag3 == 'Select':
            bound.add(node.name)
    axis_name = f'__vec_{tag}_{at}'
    fresh = {axis_name}
    for name in defined:
        for lane in range(width):
            fresh.add(f'{name}__v{lane}')
    if fresh & bound:
        return None
    for s in loop.body:
        if kind(s) == 'Load':
            lane_of[s.names[0]] = tuple(f'{s.names[0]}__v{lane}' for lane in range(width))
        elif kind(s) == 'Assign':
            lane_of[s.name] = tuple(f'{s.name}__v{lane}' for lane in range(width))
    return {'width': width, 'axis_name': axis_name, 'lane_of': lane_of, 'defined': defined}


def _rewrite(loop, tile, plan):
    width = plan['width']
    lane_of = plan['lane_of']
    var = Var(plan['axis_name'])
    scaled = BinaryExpr(op='*', left=Literal(width, dtype='int'), right=var)

    def remap(index):
        return (*index[:-1], scaled)

    body = []
    for s in loop.body:
        if kind(s) == 'Load':
            body.append(Load(names=lane_of[s.names[0]], input=s.input, index=remap(s.index), dtype=s.dtype))
        elif kind(s) == 'Write':
            widened = []
            for name in s.values:
                if name in lane_of:
                    widened.extend(lane_of[name])
                else:
                    widened.extend([name] * width)
            body.append(Write(output=s.output, index=remap(s.index), values=tuple(widened), value_dtype=s.value_dtype, atomic=False, swizzle='NONE'))
        elif kind(s) == 'Assign':
            for lane in range(width):
                args = tuple(lane_of[a][lane] if a in lane_of else a for a in s.args)
                body.append(replace(s, name=lane_of[s.name][lane], args=args))
    axis = Axis(name=plan['axis_name'], extent=Dim(loop.axis.extent.value // width))
    return StridedLoop(axis=axis, start=loop.start, step=loop.step, body=Body(tuple(body)), unroll=loop.unroll, end=None, seed=loop.seed)


def vectorize_elementwise(kernel, launch):
    threads = launch['block'][0]
    changed = False
    tiles = []
    for tag, stmt in enumerate(kernel.body):
        if kind(stmt) != 'Tile' or (stmt.block_threads or threads) != threads:
            tiles.append(stmt)
            continue
        tile = stmt
        for at, loop in enumerate(tuple(tile.body)):
            plan = _match(loop, tile, threads, tag, at, _uses_after(tile, at))
            if plan is None:
                continue
            tile = replace(tile, body=Body((*tile.body[:at], _rewrite(loop, tile, plan), *tile.body[at + 1:])))
            changed = True
        tiles.append(tile)
    if not changed:
        return None
    return replace(kernel, body=Body(tuple(tiles)))
