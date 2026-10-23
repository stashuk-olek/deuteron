import math
from typing import Any
from compiler.analysis import add_terms, canonical, constant_difference, index_variables, kind, known_multiple, substitute, uses, exprs, walk
from compiler.options import Contraction

def tensorize(kernel: Any, launch: Any, precision: str) -> Any:
    try:
        return _tensorize(kernel, launch, precision)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return ()

def _tensorize(kernel: Any, launch: Any, precision: str) -> Any:
    threads = launch['block'][0]
    if kernel.serial or threads <= 0 or threads % 32 or launch['block'][1:] != [1, 1] or launch['grid'][1:] != [1, 1]:
        return ()
    step = 16 if precision == 'bf16x3' else 8
    plans, taken = ([], set())
    for tile in kernel.body:
        if kind(tile) != 'Tile':
            continue
        plans.extend(_tensorize_tile(tile, launch, threads, step, taken))
    return tuple(plans)

def _tensorize_tile(tile: Any, launch: Any, threads: Any, step: Any, taken: Any) -> Any:
    axes = [axis.name for axis in tile.axes]
    cells = [axis.extent.value for axis in tile.axes]
    if len(tile.axes) < 2 or tile.aux_threads or len(set(axes)) != len(axes) or {'lane', 'warp'} & set(axes):
        return ()
    if tile.block_threads is not None and tile.block_threads != threads:
        return ()
    if math.prod(cells[-2:]) != threads or math.prod(cells) != threads * launch['grid'][0]:
        return ()
    if tile.raster_axes and {axes[-2], axes[-1]} & set(tile.raster_axes):
        return ()
    smem = {s.name: s for s in tile.body if kind(s) == 'Smem'}
    if any((kind(s) == 'Sync' and (s.warp or s.barrier_id or s.count is not None) for s in walk(tile.body))):
        return ()
    parents = {}
    stack = [(tile, tile.body)]
    while stack:
        node, body = stack.pop()
        for s in body:
            parents[id(s)] = node
            for attr in ('body', 'else_body', 'combine_states'):
                stack.append((s, getattr(s, attr, ())))
    scope = set(axes) | {s.axis.name for s in walk(tile.body) if kind(s) == 'StridedLoop'}
    args = {a['name']: a for a in launch['args']}
    plans = []
    for inner in walk(tile.body):
        if kind(inner) != 'Loop' or inner.seed:
            continue
        contraction = _contraction(tile, inner, axes, cells, smem, parents, scope, args, threads, step, taken)
        if contraction is not None:
            taken.update(s.name for s in contraction.loop.body if kind(s) == 'Accum')
            plans.append(contraction)
    return plans

def _contraction(tile: Any, inner: Any, axes: Any, cells: Any, smem: Any, parents: Any, scope: Any, args: Any, threads: Any, step: Any, taken: Any) -> Any:
    if inner.axis.name in set(axes) | {'lane', 'warp'}:
        return None
    depth = inner.axis.extent.value
    if depth <= 0 or depth % 8:
        return None
    outer = parents.get(id(inner))
    if outer is None or kind(outer) != 'StridedLoop' or parents.get(id(outer)) is not tile:
        return None
    if outer.axis.name in set(axes) | {inner.axis.name, 'lane', 'warp'}:
        return None
    bounds = (outer.start, outer.step) + ((outer.end,) if outer.end is not None else ())
    for bound in bounds:
        names = index_variables((bound,))
        if names is None or names - set(axes[:-2]):
            return None
    loads, assigns, pairs, accums, defined = ({}, {}, {}, {}, set())
    for s in inner.body:
        tag = kind(s)
        if tag == 'Load':
            if len(s.names) != 1 or s.input not in smem or len(s.index) != 2 or (s.dtype is not None and s.dtype.name != 'f32'):
                return None
            loads[s.names[0]] = s
            name = s.names[0]
        elif tag == 'Assign':
            if not s.op.semiring_product or s.dtype.name != 'f32' or len(s.args) != 2 or any((a not in loads for a in s.args)):
                return None
            assigns[s.name] = s
            pairs[s.name] = tuple(loads[a] for a in s.args)
            name = s.name
        elif tag == 'Accum':
            if s.value not in assigns or not assigns[s.value].op.distributes_over(s.op):
                return None
            if not (s.op.associative and s.op.identity == 0) or s.dtype.name != 'f32' or s.base not in (None, s.name):
                return None
            accums[s.name] = pairs[s.value]
            name = s.name
        else:
            return None
        if name in defined:
            return None
        defined.add(name)
    if not accums or len(assigns) != len(accums) or set(accums) & taken:
        return None
    operands = {load.input for pair in accums.values() for load in pair}
    if len(operands) != 2 or any((len(smem[b].extents) != 2 or smem[b].dtype != 'float' or smem[b].extents[1] < 4 or smem[b].extents[1] & smem[b].extents[1] - 1 for b in operands)):
        return None
    first = tile.body.index(outer)
    inits = [s for s in tile.body if kind(s) == 'Init' and s.name in accums]
    if {s.name for s in inits} != set(accums) or len(inits) != len(accums):
        return None
    if any((s.identity != 0 or s.dtype.name != 'f32' or tile.body.index(s) >= first for s in inits)):
        return None
    writes = [s for s in tile.body if kind(s) == 'Write' and set(s.values) & set(accums)]
    if not writes or any((tile.body.index(s) <= first for s in writes)) or any((set(s.values) - set(accums) for s in writes)):
        return None
    output = writes[0]
    if len(output.index) != 2 or output.swizzle != 'NONE':
        return None
    if output.output not in args or not args[output.output].get('output') or args[output.output]['dtype'] != 'f32' or len(args[output.output]['shape']) != 2:
        return None
    row_axis, col_axis = axes[-2:]
    base = tuple(substitute(index, {row_axis, col_axis}) for index in output.index)
    for index in base:
        names = index_variables((index,))
        if names is None or names - set(axes[:-2]):
            return None
    row_poly, col_poly = map(canonical, output.index)
    rows = row_poly.get(('var', row_axis), 0)
    cols = col_poly.get(('var', col_axis), 0)
    if rows <= 0 or cols <= 0:
        return None
    expected_row = add_terms(canonical(base[0]), (('var', row_axis), rows))
    expected_col = add_terms(canonical(base[1]), (('var', col_axis), cols))
    if row_poly != expected_row or col_poly != expected_col:
        return None
    tile_rows, tile_cols = (cells[-2] * rows, cells[-1] * cols)
    if tile_rows % 16 or tile_cols % 8 or tile_rows // 16 * (tile_cols // 8) > threads // 32:
        return None
    points = {}
    for w in writes:
        if w.output != output.output or w.value_dtype.name != 'f32' or w.atomic != output.atomic or w.swizzle != 'NONE' or len(w.index) != 2:
            return None
        r = constant_difference(canonical(w.index[0]), expected_row)
        c = constant_difference(canonical(w.index[1]), expected_col)
        if r is None or c is None:
            return None
        for j, name in enumerate(w.values):
            point = (r, c + j)
            if point in points or name not in accums:
                return None
            points[point] = name
    if set(points) != {(r, c) for r in range(rows) for c in range(cols)} or set(points.values()) != set(accums) or len(points) != len(accums):
        return None
    removed = {id(inner)} | {id(s) for s in walk(inner.body)}
    kept = {id(s) for s in writes} | {id(s) for s in inits}
    inner_defined = set(loads) | set(assigns) | set(accums)
    for s in walk(tile.body):
        if id(s) in removed:
            continue
        used = uses(s)
        found = exprs(s)
        if used is None or found is None:
            return None
        if id(s) not in kept and used & inner_defined:
            return None
        names = index_variables(found)
        if names is None or inner.axis.name in names:
            return None
    coordinate_names = scope - {inner.axis.name}
    for s in walk(tile.body):
        if kind(s) == 'CpAsyncCopy' and s.smem in operands:
            if s.swizzle != 'NONE' or s.nbytes != 16 or len(s.smem_index) != 2:
                return None
            lane_index = s.lane_index or (0, 0)
            if len(lane_index) != 2 or any((known_multiple(x) % 4 for x in (s.smem_index[1], lane_index[1]))):
                return None
            names = index_variables((*s.smem_index, *lane_index, *s.src_index))
            if names is None or names - coordinate_names:
                return None
    lhs_base: Any = None
    rhs_base: Any = None
    lhs_buffer: Any = None
    rhs_buffer: Any = None
    for (r, c), name in sorted(points.items()):
        oriented = []
        for load in accums[name]:
            indices = tuple(map(canonical, load.index))
            if indices[1].get(('var', inner.axis.name), 0) == 1 and ('var', inner.axis.name) not in indices[0]:
                oriented.insert(0, load)
            elif indices[0].get(('var', inner.axis.name), 0) == 1 and ('var', inner.axis.name) not in indices[1]:
                oriented.append(load)
            else:
                return None
        if len(oriented) != 2:
            return None
        lhs, rhs = oriented
        if lhs.input == rhs.input:
            return None
        if lhs_base is None:
            lhs_base = tuple(substitute(index, {row_axis, col_axis, inner.axis.name}) for index in lhs.index)
            rhs_base = tuple(substitute(index, {row_axis, col_axis, inner.axis.name}) for index in rhs.index)
            lhs_buffer, rhs_buffer = (lhs.input, rhs.input)
            for index in (*lhs_base, *rhs_base):
                names = index_variables((index,))
                if names is None or names - (set(axes[:-2]) | {outer.axis.name}):
                    return None
        if lhs.input != lhs_buffer or rhs.input != rhs_buffer:
            return None
        expected_lhs = (add_terms(canonical(lhs_base[0]), (('var', row_axis), rows), ((), r)), add_terms(canonical(lhs_base[1]), (('var', inner.axis.name), 1)))
        expected_rhs = (add_terms(canonical(rhs_base[0]), (('var', inner.axis.name), 1)), add_terms(canonical(rhs_base[1]), (('var', col_axis), cols), ((), c)))
        if tuple(map(canonical, lhs.index)) != expected_lhs or tuple(map(canonical, rhs.index)) != expected_rhs:
            return None
    if depth % step or tile_rows // 16 * (tile_cols // 8) * 32 != threads:
        return None
    return Contraction(tile, inner, outer, lhs_buffer, lhs_base, rhs_buffer, rhs_base, output.output, base, output.atomic, tile_rows, tile_cols, depth, tuple(writes))

