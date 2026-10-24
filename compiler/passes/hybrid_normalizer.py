import math
from dataclasses import replace
from typing import Any
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import WarpShuffle
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Init, Load, Select, SelectBranch
from compiler.analysis import index_variables, kind, walk
from compiler.passes.regforward import _bound_names, _no_writes_to
from compiler.passes.tailtree import _accesses, _scalar_tree
from compiler.passes.vectorize import _uses_after
from compiler.transforms import SCALAR_OPS, symbolic


def _normalizer(stmt: Any) -> Any:
    if len(stmt.state) != 2 or len(stmt.state_b) != 2 or len(set((*stmt.state, *stmt.state_b))) != 4 or stmt.dtype is not None and stmt.dtype.name != 'f32':
        return None
    env = {n: ('state', n) for n in (*stmt.state, *stmt.state_b)}
    for child in stmt.combine_states:
        if kind(child) != 'Assign' or child.op.name not in SCALAR_OPS or set(child.args) - env.keys() or child.dtype is not None and child.dtype.name != 'f32':
            return None
        env[child.name] = symbolic(child.op.name, *(env[n] for n in child.args))
    for i in range(2):
        m, s, n, t = stmt.state[i], stmt.state[1 - i], stmt.state_b[i], stmt.state_b[1 - i]
        a, b, x, y = (('state', v) for v in (m, s, n, t))
        maximum = symbolic('maximum', a, x)
        alpha = symbolic('exp', symbolic('subtract', a, maximum))
        beta = symbolic('exp', symbolic('subtract', x, maximum))
        total = symbolic('add', symbolic('multiply', b, alpha), symbolic('multiply', y, beta))
        if env[m] == maximum and env[s] == total:
            return m, s, n, t
    return None


def hybrid_normalizer(kernel: Any, launch: Any) -> Any:
    try:
        return _hybrid(kernel, launch)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def _hybrid(kernel: Any, launch: Any) -> Any:
    if kernel.serial or len(kernel.body) != 1 or kind(kernel.body[0]) != 'Tile':
        return None
    tile = kernel.body[0]
    threads = launch['block'][0]
    if not 64 <= threads <= 256 or threads & (threads - 1) or not tile.axes or tile.axes[-1].extent.value != threads:
        return None
    if math.prod(a.extent.value for a in tile.axes) != threads * launch['grid'][0] or tile.aux_threads or tile.raster_axes or tile.raster_group:
        return None
    if launch['block'][1:] != [1, 1] or launch['grid'][1:] != [1, 1] or not 0 < threads * launch['grid'][0] < 2**31 or any(not 0 < a.extent.value < 2**31 for a in tile.axes):
        return None
    if tuple(kind(s) for s in tile.body[:7]) != ('StridedLoop', 'WarpShuffle', 'Smem', 'Smem', 'Cond', 'Sync', 'TreeHalve'):
        return None
    loop, shuffle, sm0, sm1, store, sync, tree = tile.body[:7]
    if loop.end is not None or not loop.seed or loop.start != Var(tile.axes[-1].name) or loop.step != Literal(threads, dtype='int'):
        return None
    extent = loop.axis.extent.value
    width = extent // threads
    if extent % threads or width not in (1, 2, 4) or extent // 32 > 16:
        return None
    match = _normalizer(shuffle)
    if match is None or shuffle.length != 32 or tree.state != shuffle.state or tree.state_b != shuffle.state_b or tree.combine_states != shuffle.combine_states:
        return None
    if tree.dtype is not None and tree.dtype.name != 'f32':
        return None
    if tree.length != threads // 32 or tree.inner is not None or tree.tid_var != 'warp' or tree.barrier_id or tree.barrier_count is not None or sync.barrier_id or sync.count is not None or sync.warp:
        return None
    m, s, n, t = match
    loads = [q for q in loop.body if kind(q) == 'Load']
    accums = [q for q in loop.body if kind(q) == 'Accum']
    if len(loads) != 1 or len(loads[0].names) != 1 or {q.name for q in accums} != {m, s} or len(accums) != 2:
        return None
    load = loads[0]
    readonly = {a['name'] for a in launch['args'] if not a.get('output')}
    if load.input not in readonly or not _no_writes_to(tile, load.input) or not load.index or load.index[-1] != Var(loop.axis.name):
        return None
    names = index_variables(load.index[:-1])
    if names is None or names - {a.name for a in tile.axes[:-1]}:
        return None
    env = {m: ('state', m), s: ('state', s)}
    for q in loop.body:
        if getattr(q, 'dtype', None) is not None and q.dtype.name != 'f32':
            return None
        if kind(q) == 'Load':
            env[q.names[0]] = ('input', q.names[0])
        elif kind(q) == 'Let' and kind(q.value) == 'Literal':
            env[q.name] = ('const', q.value.value)
        elif kind(q) == 'Assign' and q.op.name in SCALAR_OPS and not set(q.args) - env.keys():
            env[q.name] = symbolic(q.op.name, *(env[a] for a in q.args))
        elif kind(q) == 'Accum' and q.op.name in ('add', 'maximum') and (q.base or q.name) in env and q.value in env:
            env[q.name] = symbolic(q.op.name, env[q.base or q.name], env[q.value])
        else:
            return None
    a, b, x = ('state', m), ('state', s), ('input', load.names[0])
    maximum = symbolic('maximum', a, x)
    total = symbolic('add', symbolic('multiply', b, symbolic('exp', symbolic('subtract', a, maximum))), symbolic('exp', symbolic('subtract', x, maximum)))
    if env[m] != maximum or env[s] != total or any(q.op.name != ('maximum' if q.name == m else 'add') for q in accums):
        return None
    identity = next(q.op.identity for q in accums if q.name == m)
    if not math.isfinite(identity) or next(q.op.identity for q in accums if q.name == s) != 0:
        return None
    if store.cond != BinaryExpr('==', Var('lane'), Literal(0, dtype='int')) or store.else_body or len(store.body) != 2:
        return None
    if len(tree.bufs) != 2 or len(set(tree.bufs)) != 2 or set(tree.bufs) != {sm0.name, sm1.name} or any(q.dtype != 'float' or q.extents != (tree.length,) for q in (sm0, sm1)):
        return None
    for q, buf, state in zip(store.body, tree.bufs, tree.state, strict=True):
        if kind(q) != 'Write' or q.atomic or q.swizzle != 'NONE' or q.output != buf or q.values != (state,) or q.index != (Var('warp'),):
            return None
    if len([q for q in walk(tile.body) if kind(q) == 'Smem']) != 2:
        return None
    reads, writes = _accesses(tile.body[7:])
    after = _uses_after(tile, 6)
    internal = set(env) | set(shuffle.state_b) | {q.name for q in shuffle.combine_states} | {loop.axis.name}
    if set(tree.bufs) & (reads | writes) or after is None or (internal - {m, s}) & after:
        return None
    if len(tile.body) != 9 or kind(tile.body[7]) != 'Assign' or kind(tile.body[8]) != 'StridedLoop':
        return None
    reciprocal, consumer = tile.body[7:]
    if reciprocal.op.name != 'reciprocal' or reciprocal.args != (s,) or consumer.start != loop.start or consumer.step != loop.step or consumer.end is not None or consumer.axis.extent.value != extent:
        return None
    if tuple(kind(q) for q in consumer.body) != ('Load', 'Assign', 'Assign', 'Assign', 'Write'):
        return None
    reread, delta, exponent, product, write = consumer.body
    if len(reread.names) != 1 or reread.input != load.input or reread.index != (*load.index[:-1], Var(consumer.axis.name)):
        return None
    if delta.op.name != 'subtract' or delta.args != (reread.names[0], m) or exponent.op.name != 'exp' or exponent.args != (delta.name,) or product.op.name != 'multiply' or product.args != (reciprocal.name, exponent.name):
        return None
    consumer_names = (reciprocal.name, reread.names[0], delta.name, exponent.name, product.name)
    if len(set(consumer_names)) != len(consumer_names) or set(consumer_names) & {m, s}:
        return None
    if write.atomic or write.swizzle != 'NONE' or write.index != reread.index or write.values != (product.name,):
        return None
    if any(getattr(q, 'dtype', None) is not None and q.dtype.name != 'f32' for q in (reciprocal, *consumer.body)) or write.value_dtype is not None and write.value_dtype.name != 'f32':
        return None
    args = {a['name']: a for a in launch['args']}
    shape = tuple(a.extent.value for a in tile.axes[:-1]) + (extent,)
    if load.index[:-1] != tuple(Var(a.name) for a in tile.axes[:-1]) or math.prod(shape) >= 2**31:
        return None
    for name in (load.input, write.output):
        if name not in args or args[name]['dtype'] != 'f32' or args[name].get('scalar') or tuple(args[name]['shape']) != shape:
            return None
    if not args[write.output].get('output') or set(tree.bufs) & args.keys():
        return None
    axes = [a.name for a in tile.axes] + [loop.axis.name, consumer.axis.name]
    definitions = {q.name for q in walk(tile.body) if kind(q) in ('Assign', 'Accum', 'Init', 'Let', 'Select')} | {name for q in walk(tile.body) if kind(q) == 'Load' for name in q.names}
    if len(set(axes)) != len(axes) or set(axes) & {'lane', 'warp'} or definitions & (set(axes) | {'lane', 'warp'} | args.keys() | set(tree.bufs)) or any(name.startswith(('threadIdx.', 'blockIdx.', 'blockDim.')) for name in set(axes) | definitions):
        return None
    bound = _bound_names(tile) | args.keys()
    prefix = '__hybrid'
    while any(name.startswith(prefix) for name in bound):
        prefix += '_'
    dtype = load.dtype or tree.dtype
    component = tree.state.index(s)
    shared = sm0 if sm0.name == tree.bufs[component] else sm1
    output = [shared, Init(m, float(identity), dtype=dtype)]
    cached = []
    for group in range(threads // 32):
        values = tuple(f'{prefix}_{group}_{i}' for i in range(width))
        coordinate = Var('lane') * Literal(width, dtype='int') + Literal(group * 32 * width, dtype='int')
        output.append(replace(load, names=values, index=(*load.index[:-1], coordinate)))
        output.extend(Assign(m, ElementwiseImpl('maximum'), (m, value), dtype) for value in values)
        cached.append(values)
    maximum_combine = Assign(m, ElementwiseImpl('maximum'), (m, n), dtype)
    maximum_shuffle = WarpShuffle(state=(m,), state_b=(n,), combine_states=(maximum_combine,), length=32, dtype=dtype)
    output.extend((maximum_shuffle, Init(s, 0.0, dtype=dtype)))
    values = []
    for i in range(width):
        selected, difference, value = (f'{prefix}_{tag}_{i}' for tag in ('owned', 'delta', 'exp'))
        branches = tuple(SelectBranch(value=row[i], select=BinaryExpr('==', Var('warp'), Literal(group, dtype='int'))) for group, row in enumerate(cached))
        output.extend((Select(selected, branches), Assign(difference, ElementwiseImpl('subtract'), (selected, m), dtype), Assign(value, ElementwiseImpl('exp'), (difference,), dtype), Assign(s, ElementwiseImpl('add'), (s, value), dtype)))
        values.append(value)
    sum_combine = Assign(s, ElementwiseImpl('add'), (s, t), dtype)
    output.extend((WarpShuffle(state=(s,), state_b=(t,), combine_states=(sum_combine,), length=32, dtype=dtype), replace(store, body=Body((store.body[component],))), sync))
    scalar = replace(tree, state=(s,), state_b=(t,), bufs=(tree.bufs[component],), combine_states=(sum_combine,))
    tail = _scalar_tree(scalar, 6, bound)
    if tail is None:
        return None
    output.extend((*tail, reciprocal))
    products = tuple(f'{prefix}_output_{i}' for i in range(width))
    output.extend(Assign(name, ElementwiseImpl('multiply'), (reciprocal.name, value), dtype) for name, value in zip(products, values, strict=True))
    output.append(replace(write, values=products, index=(*write.index[:-1], Var('lane') * Literal(width, dtype='int') + Var('warp') * Literal(32 * width, dtype='int'))))
    return replace(kernel, body=Body((replace(tile, body=Body(output)),))), maximum_shuffle
