import math
from dataclasses import dataclass, replace
from emmy.compiler.ir.expr import Literal
from llvmlite import ir
I1 = ir.IntType(1)
I32 = ir.IntType(32)
F32 = ir.FloatType()
VOID = ir.VoidType()
STATIC_SHARED_LIMIT = 48 * 1024
SCALAR_OPS = {'add', 'subtract', 'multiply', 'divide', 'copy', 'reciprocal', 'exp', 'exp_fast', 'sqrt', 'rsqrt', 'maximum', 'minimum', 'relu', 'negative', 'absolute'}
INDEX_OPS = {'+', '-', '*', '/', '//', '%', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>=', '&&', '||'}

@dataclass(frozen=True)
class Options:
    approximate_math: bool = True
    contraction_precision: str = 'bf16x3'
    recurrence_staging: bool = True
    reduction_sharing: bool = True
    vector_width: int = 4
    unroll: int = 0
    tree_specialization: bool = True
    coalesced_reductions: bool = True
    warp_willreturn: bool = True

@dataclass(frozen=True)
class Contraction:
    tile: object
    loop: object
    outer: object
    lhs_buffer: str
    lhs_base: tuple
    rhs_buffer: str
    rhs_base: tuple
    output_buffer: str
    output_base: tuple
    atomic: bool
    rows: int
    cols: int
    depth: int
    writes: tuple

@dataclass
class Recurrence:
    tile: object
    loop: object
    prefix: tuple
    suffix: tuple
    reductions: tuple
    invariant: tuple
    varying: tuple
    scale: str
    width: int
    extent: int
    components: tuple
    gather: tuple
    identity: float
    tag: int

@dataclass(frozen=True)
class Hints:
    contractions: tuple = ()
    recurrences: tuple = ()

def kind(value):
    return type(value).__name__

def walk(body):
    for statement in body:
        yield statement
        for attr in ('body', 'else_body', 'combine_states'):
            yield from walk(getattr(statement, attr, ()))

def free_variables(expression):
    if isinstance(expression, int):
        return set()
    tag = kind(expression)
    if tag == 'Var':
        return {expression.name}
    if tag in ('Literal', 'Builtin'):
        return {expression.name} if tag == 'Builtin' else set()
    if tag == 'CastExpr':
        return free_variables(expression.expr)
    if tag == 'FlatIndex':
        return index_variables(expression.index)
    if tag == 'FuncCallExpr':
        return index_variables(expression.args)
    if tag == 'BinaryExpr':
        if expression.op not in INDEX_OPS:
            return None
        children = (expression.left, expression.right)
    elif tag == 'TernaryExpr':
        children = (expression.cond, expression.if_true, expression.if_false)
    else:
        return None
    return index_variables(children)

def index_variables(indices):
    result = set()
    for index in indices:
        names = free_variables(index)
        if names is None:
            return None
        result.update(names)
    return result

def substitute(expression, names):
    if isinstance(expression, int) or kind(expression) == 'Literal':
        return expression
    if kind(expression) == 'Var':
        return Literal(0, dtype='int') if expression.name in names else expression
    fields = {'BinaryExpr': ('left', 'right'), 'TernaryExpr': ('cond', 'if_true', 'if_false')}.get(kind(expression))
    if fields:
        return replace(expression, **{name: substitute(getattr(expression, name), names) for name in fields})
    raise ValueError('unsupported coordinate expression')

def canonical(expression):

    def frozen(value):
        return tuple(sorted(value.items(), key=repr))
    if isinstance(expression, int):
        return {(): expression} if expression else {}
    if kind(expression) == 'Literal':
        if not isinstance(expression.value, int):
            raise ValueError('noninteger coordinate')
        return canonical(expression.value)
    if kind(expression) == 'Var':
        return {('var', expression.name): 1}
    if kind(expression) == 'TernaryExpr':
        return {('select', *(frozen(canonical(getattr(expression, f))) for f in ('cond', 'if_true', 'if_false'))): 1}
    if kind(expression) != 'BinaryExpr':
        raise ValueError('unsupported coordinate')
    left, right = (canonical(expression.left), canonical(expression.right))
    if expression.op in ('+', '-'):
        return add_terms(left, *((term, coefficient * (1 if expression.op == '+' else -1)) for term, coefficient in right.items()))
    if expression.op == '*':
        if not left or not right:
            return {}
        for scalar, polynomial in ((left, right), (right, left)):
            if set(scalar) <= {()}:
                return {term: coefficient * scalar[()] for term, coefficient in polynomial.items()}
        raise ValueError('non-affine coordinate product')
    if expression.op in ('/', '//', '%', '<', '<=', '>', '>=', '==', '!=', '&', '|', '^', '<<', '>>'):
        if expression.op in ('/', '//', '%') and (not right):
            raise ValueError('zero divisor')
        if expression.op in ('/', '//') and right == {(): 1}:
            return left
        if not left and expression.op in ('/', '//', '%'):
            return {}
        return {(expression.op, frozen(left), frozen(right)): 1}
    raise ValueError('unsupported coordinate operator')

def add_terms(base, *terms):
    result = dict(base)
    for term, coefficient in terms:
        result[term] = result.get(term, 0) + coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}

def difference(left, right):
    return add_terms(left, *((term, -coefficient) for term, coefficient in right.items()))

def constant_difference(left, right):
    value = difference(left, right)
    return value.get((), 0) if set(value) <= {()} else None

def known_multiple(expression):
    if isinstance(expression, int):
        return abs(expression)
    if kind(expression) == 'Literal':
        return abs(expression.value) if isinstance(expression.value, int) else 1
    if kind(expression) != 'BinaryExpr':
        return 1
    left, right = (known_multiple(expression.left), known_multiple(expression.right))
    if expression.op in ('+', '-'):
        return math.gcd(left, right)
    if expression.op == '*':
        return left * right
    if expression.op == '%':
        return math.gcd(left, right)
    if expression.op in ('/', '//'):
        divisor = canonical(expression.right)
        if set(divisor) == {()} and divisor[()] > 0 and (left % divisor[()] == 0):
            return left // divisor[()]
    return 1

def integer(value):
    return ir.Constant(I32, int(value))

def reciprocal_sqrt_source(statement, previous):
    if type(statement).__name__ == 'Assign' and statement.op.name == 'reciprocal' and (type(previous).__name__ == 'Assign') and (previous.op.name == 'sqrt') and (len(previous.args) == 1) and (statement.args == (previous.name,)) and (previous.name != previous.args[0]):
        return previous.args[0]
    return None

def sharing_width(loop, tile, uniform_axes):
    if type(loop).__name__ != 'Loop' or len(loop.body) != 4:
        return 1
    if (tile.block_threads or 256) % 32:
        return 1
    if tile.raster_group and tile.raster_axes:
        for index, axis in enumerate(tile.axes):
            if axis.name in tile.raster_axes and axis.extent.value > 1:
                stride = math.prod((a.extent.value for a in tile.axes[index + 1:]))
                if stride % 32:
                    return 1
    left, right, product, accum = loop.body
    if tuple((type(s).__name__ for s in loop.body)) != ('Load', 'Load', 'Assign', 'Accum'):
        return 1
    if len(left.names) != 1 or len(right.names) != 1 or not product.op.semiring_product:
        return 1
    if set(product.args) != {left.names[0], right.names[0]}:
        return 1
    if not (accum.op.associative and accum.op.identity == 0) or not product.op.distributes_over(accum.op) or accum.base is not None or accum.value != product.name:
        return 1
    used = set()
    for load in (left, right):
        names = index_variables(load.index)
        if names is None:
            return 1
        used.update(names)
    used.discard(loop.axis.name)
    if used - {axis.name for axis in tile.axes} - uniform_axes:
        return 1
    width = 1
    for axis in reversed(tile.axes):
        if axis.name in used and axis.extent.value != 1:
            break
        width *= axis.extent.value
        if width >= 32:
            if width % 32:
                return 1
            width = 32
            break
    if width & width - 1 or loop.axis.extent.value < width:
        return 1
    return width

def plain_reduction(loop):
    if type(loop).__name__ != 'StridedLoop':
        return False
    if any((type(s).__name__ not in ('Load', 'Assign', 'Accum', 'Let') for s in loop.body)):
        return False
    accumulators = [s for s in loop.body if type(s).__name__ == 'Accum']
    return len(accumulators) == 1 and accumulators[0].op.associative and accumulators[0].op.identity == 0 and (accumulators[0].base is None)

def striped_reduction(loop, tile, threads):
    if not plain_reduction(loop) or len(loop.body) not in (2, 3) or (not loop.seed):
        return False
    load, accum = (loop.body[0], loop.body[-1])
    if type(load).__name__ != 'Load' or len(load.names) != 1:
        return False
    if len(loop.body) == 3:
        square = loop.body[1]
        if type(square).__name__ != 'Assign' or not square.op.semiring_product:
            return False
        if square.args != (load.names[0], load.names[0]) or accum.value != square.name:
            return False
    elif accum.value != load.names[0]:
        return False
    if type(loop.start).__name__ != 'Var' or loop.start.name != tile.axes[-1].name:
        return False
    if type(loop.step).__name__ != 'Literal' or loop.step.value != threads or tile.axes[-1].extent.value != threads:
        return False
    if loop.end is not None or loop.axis.extent.value % (threads * 4):
        return False
    if not load.index or type(load.index[-1]).__name__ != 'Var' or load.index[-1].name != loop.axis.name:
        return False
    row_axes = {axis.name for axis in tile.axes[:-1]}
    for index in load.index[:-1]:
        names = index_variables((index,))
        if names is None or names - row_axes:
            return False
    for index, child in enumerate(tile.body[:-1]):
        if child is loop:
            combine = tile.body[index + 1]
            if type(combine).__name__ != 'WarpShuffle' or combine.state != (accum.name,) or combine.length != 32:
                return False
            return any((type(s).__name__ == 'TreeHalve' and s.state == (accum.name,) and (s.length == threads // 32) and (s.tid_var == 'warp') for s in tile.body[index + 2:]))
    return False

def default_unroll(loop, tile, uniform_axes):
    if plain_reduction(loop):
        return 32
    if tile is not None:
        inner = sum((sharing_width(child, tile, uniform_axes | {loop.axis.name}) > 1 for child in loop.body))
        if inner == 1:
            return 32
    return 8

def shared_layout(columns, role, mode):
    if role == 'row':
        shift = 3 if mode == 'bf16x3' else 2
        return (shift, min(3 if mode == 'bf16x3' else 7, columns - 1 >> shift), False)
    return (3, min(3, columns - 1 >> 3), mode == 'bf16x3')

def uses(statement):
    tag = kind(statement)
    if tag == 'Write':
        return set(statement.values)
    if tag == 'Assign':
        return set(statement.args)
    if tag == 'Accum':
        used = {statement.value}
        if statement.base is not None and statement.base != statement.name:
            used.add(statement.base)
        return used
    if tag in ('WarpShuffle', 'TreeHalve'):
        return set(statement.state) | set(statement.state_b)
    if tag == 'Select':
        return {branch.value for branch in statement.branches}
    if tag == 'Carry':
        return {statement.value}
    if tag == 'Pre':
        return {statement.carrier}
    if tag in ('Init', 'Load', 'Let', 'Cond', 'Loop', 'StridedLoop', 'Sync', 'Smem', 'CpAsyncCopy', 'CpAsyncCommit', 'CpAsyncWait', 'Tile'):
        return set()
    return None

def exprs(statement):
    tag = kind(statement)
    if tag in ('Load', 'Write'):
        return tuple(statement.index)
    if tag == 'Let':
        return (statement.value,)
    if tag == 'Cond':
        return (statement.cond,)
    if tag == 'StridedLoop':
        return (statement.start, statement.step) + ((statement.end,) if statement.end is not None else ())
    if tag == 'CpAsyncCopy':
        return tuple(statement.smem_index) + tuple(statement.lane_index or ()) + tuple(statement.src_index)
    if tag == 'Select':
        return tuple(branch.select for branch in statement.branches)
    if tag in ('Carry', 'Pre'):
        return tuple(statement.index)
    if tag in ('Assign', 'Accum', 'Init', 'Loop', 'Sync', 'Smem', 'CpAsyncCommit', 'CpAsyncWait', 'WarpShuffle', 'TreeHalve', 'Tile'):
        return ()
    return None

def tensorize(kernel, launch, precision):
    try:
        return _tensorize(kernel, launch, precision)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return ()

def _tensorize(kernel, launch, precision):
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

def _tensorize_tile(tile, launch, threads, step, taken):
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

def _contraction(tile, inner, axes, cells, smem, parents, scope, args, threads, step, taken):
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
    lhs_base = rhs_base = lhs_buffer = rhs_buffer = None
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

def stage_recurrences(kernel, launch):
    readonly = {a['name'] for a in launch['args'] if not a.get('output')}
    found = []
    for tag, tile in enumerate(kernel.body):
        if kind(tile) != 'Tile':
            continue
        plan = _stage_tile(tile, launch, readonly, tag)
        if plan is not None:
            found.append(plan)
    return tuple(found)

def _stage_tile(tile, launch, readonly, tag):
    if not tile.block_threads or tile.block_threads % 32 or math.prod((a.extent.value for a in tile.axes)) % 32:
        return None
    staged = []
    for at, outer in enumerate(tile.body):
        if kind(outer) != 'Loop' or not outer.seed or not any((kind(s) == 'Loop' for s in outer.body)):
            continue
        plan = _recurrence(tile, outer, at, launch, readonly, tag)
        if plan is not None:
            staged.append(plan)
    return staged[0] if len(staged) == 1 else None

def symbolic(op, *args):
    if op == 'copy':
        return args[0]
    if op in ('multiply', 'add', 'maximum'):
        if op == 'multiply':
            if args[0] == ('const', 1.0):
                return args[1]
            if args[1] == ('const', 1.0):
                return args[0]
        args = tuple(sorted(args, key=repr))
    return (op, *args)

def uniform_prefix_values(prefix, tile, width):
    uniform_axes = {'warp'}
    for index, axis in enumerate(tile.axes):
        stride = math.prod((a.extent.value for a in tile.axes[index + 1:]))
        if axis.extent.value == 1 or stride % width == 0:
            uniform_axes.add(axis.name)
    values = {}
    for statement in prefix:
        tag = kind(statement)
        if tag == 'Load':
            names = index_variables(statement.index)
            if names is None:
                return None
            uniform = not names - uniform_axes
            values.update(((name, uniform) for name in statement.names))
        elif tag == 'Let':
            names = free_variables(statement.value)
            if names is None:
                return None
            known_uniform = uniform_axes | {name for name, uniform in values.items() if uniform}
            values[statement.name] = not names - known_uniform
        elif tag == 'Assign':
            if statement.op.name not in SCALAR_OPS:
                return None
            values[statement.name] = all((values.get(name, False) for name in statement.args))
        else:
            return None
    return values

def _recurrence(tile, outer, at, launch, readonly, tag):
    prefix, suffix = (tile.body[:at], tile.body[at + 1:])
    if any((kind(s) not in ('Load', 'Let', 'Assign') for s in prefix)):
        return None
    reductions = tuple(s for s in outer.body if kind(s) == 'Loop')
    if not reductions:
        return None
    widths = [sharing_width(d, tile, {outer.axis.name}) for d in reductions]
    if min(widths) <= 1 or min(widths) != max(widths):
        return None
    width = widths[0]
    extent = outer.axis.extent.value
    if extent <= 0 or extent % width or extent // width > 8:
        return None
    if len({(d.axis.name, d.axis.extent.value) for d in reductions}) != 1:
        return None
    if reductions[0].axis.name == outer.axis.name:
        return None
    shapes = {(d.body[2].op.name, d.body[3].op.name) for d in reductions}
    if len(shapes) != 1:
        return None
    product_name, combine_name = next(iter(shapes))
    product_op = reductions[0].body[2].op
    dot_names = set()
    for reduction in reductions:
        left, right, product, accumulator = reduction.body
        names = {left.names[0], right.names[0], product.name, accumulator.name}
        if len(names) != 4 or names & dot_names or accumulator.op.identity != 0:
            return None
        dot_names.update(names)
    invariant, varying = ([], [])
    for reduction in reductions:
        first, second = reduction.body[:2]
        first_names, second_names = (index_variables(first.index), index_variables(second.index))
        if first_names is None or second_names is None:
            return None
        if outer.axis.name in first_names:
            first, second = (second, first)
            first_names, second_names = (second_names, first_names)
        if outer.axis.name in first_names:
            return None
        if outer.axis.name not in second_names:
            return None
        invariant.append(first)
        varying.append(second)
    if any(((k.input, k.index) != (varying[0].input, varying[0].index) for k in varying)):
        return None
    fixed = [load.names[0] for load in invariant]
    shared = {load.names[0] for load in varying}
    if len(set(fixed)) != len(fixed) or set(fixed) & shared:
        return None
    states = {s.name: s for s in outer.body if kind(s) == 'Accum'}
    if dot_names & set(states):
        return None
    if len(states) != sum((kind(s) == 'Accum' for s in outer.body)):
        return None

    def mass(op):
        return op.associative and op.identity == 0 and product_op.distributes_over(op)
    if any((not s.op.selecting and not mass(s.op) for s in states.values())):
        return None
    env = {n: ('state', n) for n in states}
    external = set()
    inputs = {}
    gather = []

    def get(n):
        if n not in env:
            external.add(n)
            env[n] = ('external', n)
        return env[n]
    for s in outer.body:
        form = kind(s)
        if form == 'Loop':
            env[s.body[-1].name] = ('reduction', reductions.index(s))
        elif form == 'Load':
            gather.append(s)
            for n in s.names:
                env[n] = ('input', n)
                inputs[n] = s
        elif form == 'Let':
            if kind(s.value) != 'Literal':
                return None
            env[s.name] = ('const', float(s.value.value))
        elif form == 'Assign':
            if s.op.name not in SCALAR_OPS:
                return None
            env[s.name] = symbolic(s.op.name, *(get(a) for a in s.args))
        elif form == 'Accum':
            env[s.name] = symbolic(s.op.name, get(s.base or s.name), get(s.value))
        else:
            return None
    gathered = [name for load in gather for name in load.names]
    if len(set(gathered)) != len(gathered) or set(gathered) & set(states):
        return None
    components = []
    covered = set()
    scales = set()
    identities = set()
    for m, stmt in states.items():
        if not stmt.op.selecting:
            continue
        expression = env[m]
        if expression[0] != stmt.op.name or ('state', m) not in expression[1:]:
            return None
        scores = [e for e in expression[1:] if e != ('state', m)]
        if len(expression) != 3 or len(scores) != 1:
            return None
        score = scores[0]
        if score[0] != product_name or len(score) != 3:
            return None
        dot = next((a for a in score[1:] if isinstance(a, tuple) and len(a) == 2 and a[0] == 'reduction'), None)
        scale = next((a for a in score[1:] if isinstance(a, tuple) and len(a) == 2 and a[0] == 'external'), None)
        if dot is None or scale is None:
            return None
        scales.add(scale[1])
        part = dot[1]
        identities.add(stmt.op.identity)
        alpha = symbolic('exp', symbolic('subtract', ('state', m), expression))
        beta = symbolic('exp', symbolic('subtract', score, expression))
        norms = []
        weights = []
        for name, state in states.items():
            if not mass(state.op):
                continue
            carried = symbolic(product_name, ('state', name), alpha)
            if env[name] == symbolic(combine_name, carried, beta):
                norms.append(name)
            for name_in in inputs:
                if env[name] == symbolic(combine_name, carried, symbolic(product_name, ('input', name_in), beta)):
                    weights.append((name, name_in))
        if len(norms) != 1 or len(weights) != 1:
            return None
        components.append((part, m, norms[0], weights[0][0], weights[0][1]))
        covered.update((m, norms[0], weights[0][0]))
    if covered != set(states) or len(scales) != 1 or len(identities) != 1:
        return None
    if math.isnan(next(iter(identities))):
        return None
    definitions = set()
    for statement in walk(outer.body):
        form = kind(statement)
        if form == 'Load':
            definitions.update(statement.names)
        elif form in ('Assign', 'Accum', 'Let', 'Init'):
            definitions.add(statement.name)
    if external & definitions:
        return None
    scale = next(iter(scales))
    prefix_values = uniform_prefix_values(prefix, tile, width)
    if prefix_values is None or not prefix_values.get(scale, False):
        return None
    preserved = {axis.name for axis in tile.axes} | {'lane', 'warp'}
    counters = preserved | {outer.axis.name, reductions[0].axis.name}
    if definitions & counters or set(prefix_values) & counters:
        return None
    final = {name: ('final', name) for name in states}
    written = set()
    expected = {symbolic(product_name, ('final', y), symbolic('reciprocal', ('final', n))): (part, y) for part, m, n, y, name_in in components}
    for s in suffix:
        if kind(s) == 'Assign':
            if s.op.name not in SCALAR_OPS or s.name in counters or any((a not in final for a in s.args)):
                return None
            final[s.name] = symbolic(s.op.name, *(final[a] for a in s.args))
        elif kind(s) == 'Write':
            if s.atomic or s.swizzle != 'NONE':
                return None
            coordinates = index_variables(s.index)
            if coordinates is None or coordinates - preserved:
                return None
            for name in s.values:
                if final.get(name) not in expected:
                    return None
                written.add(expected[final[name]][1])
        else:
            return None
    if written != {c[3] for c in components}:
        return None
    if {c[0] for c in components} != set(range(len(reductions))):
        return None
    allowed = {a.name for a in tile.axes} | {outer.axis.name}
    for load in gather:
        coordinates = index_variables(load.index)
        if coordinates is None or coordinates - allowed:
            return None
    if launch['block'][0] != tile.block_threads:
        return None
    prefix_loads = tuple(s for s in prefix if kind(s) == 'Load')
    if any((load.input not in readonly for load in (*prefix_loads, *invariant, *varying, *gather))):
        return None
    parts = len(reductions)
    count = launch['block'][0] // width * parts * extent
    if count * 4 > STATIC_SHARED_LIMIT or parts * (extent // width) > 32:
        return None
    return Recurrence(tile, outer, prefix, suffix, reductions, tuple(invariant), tuple(varying), scale, width, extent, tuple(components), tuple(gather), next(iter(identities)), tag)

class RecurrenceEmitter:

    def __init__(self, lowering, plan):
        buffers_key = f'__staged_{plan.tag}'
        if buffers_key in lowering.buffers or f'staged_weights_{plan.tag}' in lowering.module.globals:
            raise ValueError('staged recurrence scratch already allocated')
        if any((name.startswith(f'__score_{plan.tag}_') or name.startswith(f'__normalizer_{plan.tag}_') for name in lowering.slots)):
            raise ValueError('staged recurrence slots already allocated')
        self.lowering = lowering
        self.plan = plan
        self.scores = f'__score_{plan.tag}'
        self.normalizer = f'__normalizer_{plan.tag}'
        self.staged = buffers_key
        parts = len(plan.reductions)
        rounds = plan.extent // plan.width
        for part in range(parts):
            for key in range(rounds):
                lowering.allocate(f'{self.scores}_{part}_{key}', F32)
            lowering.allocate(f'{self.normalizer}_{part}', F32)
        count = lowering.launch['block'][0] // plan.width * parts * plan.extent
        scratch = ir.GlobalVariable(lowering.module, ir.ArrayType(F32, count), f'staged_weights_{plan.tag}', addrspace=3)
        scratch.linkage = 'internal'
        scratch.initializer = ir.Constant(scratch.value_type, ir.Undefined)
        scratch.align = 16
        lowering.buffers[self.staged] = lowering.builder.gep(scratch, [integer(0), integer(0)])
        lowering.shapes[self.staged] = (count,)

    def loop_n(self, name, extent, callback, unroll=8):
        lowering = self.lowering
        lowering.put(name, integer(0))
        header = lowering.function.append_basic_block(name + '.phase.test')
        body = lowering.function.append_basic_block(name + '.phase.body')
        end = lowering.function.append_basic_block(name + '.phase.end')
        lowering.builder.branch(header)
        lowering.builder.position_at_end(header)
        lowering.builder.cbranch(lowering.builder.icmp_unsigned('<', lowering.get(name), integer(extent)), body, end)
        lowering.builder.position_at_end(body)
        callback()
        lowering.put(name, lowering.builder.add(lowering.get(name), integer(1)))
        latch = lowering.builder.branch(header)
        hint = lowering.module.add_metadata(['llvm.loop.unroll.count', integer(lowering.options.unroll or unroll)])
        md = ir.MDValue(lowering.module, (), name=str(len(lowering.module.metadata)))
        hints = [hint]
        if name == self.plan.loop.axis.name:
            hints.extend([lowering.module.add_metadata(['llvm.loop.vectorize.enable', ir.Constant(I1, 1)]), lowering.module.add_metadata(['llvm.loop.vectorize.width', integer(8)]), lowering.module.add_metadata(['llvm.loop.interleave.count', integer(1)])])
        md.operands = (md, *hints)
        latch.set_metadata('llvm.loop', md)
        lowering.builder.position_at_end(end)

    def reduce(self, value, width, op):
        lowering = self.lowering
        distance = width // 2
        while distance:
            bits = lowering.builder.bitcast(value, I32)
            other = lowering.intrinsic('llvm.nvvm.shfl.sync.bfly.i32', I32, [integer(-1), bits, integer(distance), integer(31)])
            value = lowering.arithmetic(op, [value, lowering.builder.bitcast(other, F32)])
            distance //= 2
        return value

    def emit(self):
        lowering = self.lowering
        p = self.plan
        lowering.body(p.prefix)
        parts = len(p.reductions)
        rounds = p.extent // p.width
        keyaxis = p.loop.axis.name
        kaxis = p.reductions[0].axis.name
        lane = lowering.builder.urem(lowering.get('lane'), integer(p.width))
        group = lowering.builder.udiv(lowering.tid, integer(p.width))
        base = lowering.builder.mul(group, integer(parts * p.extent))
        scale = lowering.get(p.scale)
        for part in range(parts):
            for r in range(rounds):
                lowering.put(f'{self.scores}_{part}_{r}', ir.Constant(F32, 0.0))

        def score_step():
            for load in p.invariant:
                lowering.body((load,))
            for r in range(rounds):
                lowering.put(keyaxis, lowering.builder.add(lane, integer(r * p.width)))
                lowering.body((p.varying[0],))
                kval = lowering.get(p.varying[0].names[0])
                for part, load in enumerate(p.invariant):
                    name = f'{self.scores}_{part}_{r}'
                    product = lowering.arithmetic('multiply', [lowering.get(load.names[0]), kval])
                    lowering.put(name, lowering.arithmetic('add', [lowering.get(name), product]))
        self.loop_n(kaxis, p.reductions[0].axis.extent.value, score_step, unroll=32 if len(p.reductions) == 1 else 8)
        for part in range(parts):
            scores = [lowering.arithmetic('multiply', [lowering.get(f'{self.scores}_{part}_{r}'), scale]) for r in range(rounds)]
            maximum = ir.Constant(F32, p.identity)
            for score in scores:
                maximum = lowering.arithmetic('maximum', [maximum, score])
            maximum = self.reduce(maximum, p.width, 'maximum')
            mass = ir.Constant(F32, 0.0)
            for r, score in enumerate(scores):
                weight = lowering.arithmetic('exp', [lowering.arithmetic('subtract', [score, maximum])])
                offset = lowering.builder.add(base, lowering.builder.add(lane, integer(part * p.extent + r * p.width)))
                lowering.builder.store(weight, lowering.address(self.staged, (offset,)), align=4)
                mass = lowering.arithmetic('add', [mass, weight])
            mass = self.reduce(mass, p.width, 'add')
            lowering.put(f'{self.normalizer}_{part}', mass)
        lowering.intrinsic('llvm.nvvm.bar.warp.sync', VOID, [integer(-1)])
        for _part, _m, _n, y, _name_in in p.components:
            lowering.put(y, ir.Constant(F32, 0.0))

        def value_step():
            lowering.body(p.gather)
            weights = []
            for part in range(parts):
                offset = lowering.builder.add(base, lowering.builder.add(lowering.get(keyaxis), integer(part * p.extent)))
                weights.append(lowering.builder.load(lowering.address(self.staged, (offset,)), align=4))
            for part, _m, _n, y, name_in in p.components:
                product = lowering.arithmetic('multiply', [weights[part], lowering.get(name_in)])
                lowering.put(y, lowering.builder.fadd(lowering.get(y), product, flags=('reassoc', 'nsz', 'contract')))
        self.loop_n(keyaxis, p.extent, value_step, unroll=max(1, 16 // len(p.reductions)))
        for part, _m, n, _y, _name_in in p.components:
            lowering.put(n, lowering.get(f'{self.normalizer}_{part}'))
        lowering.body(p.suffix)
        lowering.striped = True

def run_passes(kernel, launch, options):
    if options.contraction_precision not in ('f32', 'tf32x3', 'bf16x3'):
        raise ValueError(f'unsupported contraction precision {options.contraction_precision}')
    contractions = tensorize(kernel, launch, options.contraction_precision) if options.contraction_precision != 'f32' else ()
    recurrences = stage_recurrences(kernel, launch) if options.recurrence_staging else ()
    return Hints(contractions, recurrences)
