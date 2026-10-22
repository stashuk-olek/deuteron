import math
from dataclasses import dataclass, replace
from emmy.compiler.ir.expr import Literal

def kind(value):
    return type(value).__name__

def walk(statements):
    for statement in statements:
        yield statement
        for attr in ('body', 'else_body', 'combine_states'):
            yield from walk(getattr(statement, attr, ()))

def walk_with_parent(parent):
    for attr in ('body', 'else_body', 'combine_states'):
        for statement in getattr(parent, attr, ()):
            yield (statement, parent)
            yield from walk_with_parent(statement)

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

def variables(value):
    if not isinstance(value, tuple):
        return set()
    if len(value) == 2 and value[0] == 'var':
        return {value[1]}
    return set().union(*(variables(item) for item in value))

def names_in(expression):
    return set().union(*(variables(term) for term in canonical(expression)))

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

def add_terms(base, *terms):
    result = dict(base)
    for term, coefficient in terms:
        result[term] = result.get(term, 0) + coefficient
    return {term: coefficient for term, coefficient in result.items() if coefficient}

@dataclass(frozen=True)
class MatmulPlan:
    tile: object
    inner_loop: object
    outer_loop: object
    a_buffer: str
    b_buffer: str
    a_base: tuple
    b_base: tuple
    output_buffer: str
    output_base: tuple
    atomic: bool
    tile_m: int
    tile_n: int
    tile_k: int
    output_writes: tuple

def recognize_matmul(kernel, launch):
    try:
        return _recognize_matmul(kernel, launch)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None

def _recognize_matmul(kernel, launch):
    if len(kernel.body) != 1 or kind(kernel.body[0]) != 'Tile' or kernel.serial:
        return None
    tile = kernel.body[0]
    threads = launch['block'][0]
    if len(tile.axes) < 2 or tile.aux_threads or threads <= 0 or threads % 32 or (tile.block_threads is not None and tile.block_threads != threads) or (launch['block'][1:] != [1, 1]) or (launch['grid'][1:] != [1, 1]) or (math.prod((axis.extent.value for axis in tile.axes[-2:])) != threads) or (math.prod((axis.extent.value for axis in tile.axes)) != threads * launch['grid'][0]):
        return None
    axis_names = [axis.name for axis in tile.axes]
    if len(set(axis_names)) != len(axis_names) or {'lane', 'warp'} & set(axis_names):
        return None
    row_axis, col_axis = axis_names[-2:]
    local = {row_axis, col_axis}
    if tile.raster_axes and local & set(tile.raster_axes):
        return None
    allowed = set('Init Smem CpAsyncCopy CpAsyncCommit CpAsyncWait Sync StridedLoop Loop Write Load Assign Accum'.split())
    all_statements = list(walk(tile.body))
    if any((kind(statement) not in allowed for statement in all_statements)):
        return None
    loops = [statement for statement in all_statements if kind(statement) == 'Loop']
    if len(loops) != 1:
        return None
    inner = loops[0]
    if inner.seed or inner.axis.name in set(axis_names) | {'lane', 'warp'}:
        return None
    k_axis = inner.axis.name
    tile_k = inner.axis.extent.value
    if tile_k <= 0 or tile_k % 8:
        return None
    parents = [statement for statement in tile.body if kind(statement) == 'StridedLoop' and any((child is inner for child in statement.body))]
    if len(parents) != 1:
        return None
    outer = parents[0]
    if outer.axis.name in set(axis_names) | {k_axis, 'lane', 'warp'}:
        return None
    outer_bounds = (outer.start, outer.step) + ((outer.end,) if outer.end is not None else ())
    if any((names_in(bound) - set(axis_names[:-2]) for bound in outer_bounds)):
        return None
    coordinate_names = (set(axis_names) | {s.axis.name for s in all_statements if kind(s) == 'StridedLoop'}) - {k_axis}
    for statement, parent in walk_with_parent(tile):
        if kind(statement) in ('Load', 'Assign', 'Accum') and parent is not inner:
            return None
        if kind(statement) == 'Write' and parent is not tile:
            return None
        if kind(statement) in ('Init', 'Smem') and parent is not tile:
            return None
        if kind(statement) == 'Sync' and (statement.warp or statement.barrier_id or statement.count is not None):
            return None
        if kind(statement) == 'CpAsyncCopy':
            if statement.swizzle != 'NONE' or statement.nbytes != 16 or len(statement.smem_index) != 2:
                return None
            lane_index = statement.lane_index or (0, 0)
            if len(lane_index) != 2 or any((known_multiple(x) % 4 for x in (statement.smem_index[1], lane_index[1]))):
                return None
            if any((names_in(x) - coordinate_names for x in (*statement.smem_index, *lane_index, *statement.src_index))):
                return None
        if kind(statement) == 'StridedLoop':
            bounds = (statement.start, statement.step) + ((statement.end,) if statement.end is not None else ())
            if any((names_in(bound) - coordinate_names for bound in bounds)):
                return None
    smem = {statement.name: statement for statement in tile.body if kind(statement) == 'Smem'}
    if len(smem) != 2 or any((len(s.extents) != 2 or s.dtype != 'float' or s.extents[1] < 4 or s.extents[1] & s.extents[1] - 1 for s in smem.values())):
        return None
    loads, products, contractions = ({}, {}, {})
    defined = set()
    for statement in inner.body:
        tag = kind(statement)
        if tag == 'Load':
            if len(statement.names) != 1 or statement.input not in smem or len(statement.index) != 2 or (statement.dtype is not None and statement.dtype.name != 'f32'):
                return None
            name = statement.names[0]
            loads[name] = statement
        elif tag == 'Assign':
            if statement.op.name != 'multiply' or statement.dtype.name != 'f32' or len(statement.args) != 2 or any((name not in loads for name in statement.args)):
                return None
            name = statement.name
            products[name] = tuple((loads[arg] for arg in statement.args))
        elif tag == 'Accum':
            if statement.op.name != 'add' or statement.dtype.name != 'f32' or statement.op.identity != 0 or (statement.base not in (None, statement.name)) or (statement.value not in products):
                return None
            name = statement.name
            contractions[name] = products[statement.value]
        else:
            return None
        if name in defined:
            return None
        defined.add(name)
    if not contractions or len(products) != len(contractions):
        return None
    inits = [statement for statement in tile.body if kind(statement) == 'Init']
    if {s.name for s in inits} != set(contractions) or len(inits) != len(contractions) or any((s.identity != 0 or s.dtype.name != 'f32' for s in inits)):
        return None
    first_outer = tile.body.index(outer)
    if any((tile.body.index(s) >= first_outer for s in inits)):
        return None
    writes = tuple((statement for statement in tile.body if kind(statement) == 'Write'))
    if not writes or any((tile.body.index(s) <= first_outer for s in writes)):
        return None
    output = writes[0]
    if len(output.index) != 2 or output.swizzle != 'NONE':
        return None
    args = {a['name']: a for a in launch['args']}
    if output.output not in args or not args[output.output].get('output') or args[output.output]['dtype'] != 'f32' or (len(args[output.output]['shape']) != 2):
        return None
    base = tuple((substitute(index, local) for index in output.index))
    if any((names_in(index) - set(axis_names[:-2]) for index in base)):
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
    tile_m, tile_n = (tile.axes[-2].extent.value * rows, tile.axes[-1].extent.value * cols)
    if tile_m % 16 or tile_n % 8 or tile_m // 16 * (tile_n // 8) > threads // 32:
        return None
    points = {}
    for write in writes:
        if write.output != output.output or write.value_dtype.name != 'f32' or write.atomic != output.atomic or (write.swizzle != 'NONE') or (len(write.index) != 2):
            return None
        r = constant_difference(canonical(write.index[0]), expected_row)
        c = constant_difference(canonical(write.index[1]), expected_col)
        if r is None or c is None:
            return None
        for j, name in enumerate(write.values):
            point = (r, c + j)
            if point in points or name not in contractions:
                return None
            points[point] = name
    if set(points) != {(r, c) for r in range(rows) for c in range(cols)} or set(points.values()) != set(contractions) or len(points) != len(contractions):
        return None
    a_base = b_base = a_buffer = b_buffer = None
    for (r, c), name in sorted(points.items()):
        pair = contractions[name]
        oriented = []
        for load in pair:
            indices = tuple(map(canonical, load.index))
            if indices[1].get(('var', k_axis), 0) == 1 and ('var', k_axis) not in indices[0]:
                oriented.insert(0, load)
            elif indices[0].get(('var', k_axis), 0) == 1 and ('var', k_axis) not in indices[1]:
                oriented.append(load)
            else:
                return None
        if len(oriented) != 2:
            return None
        a, b = oriented
        if a.input == b.input:
            return None
        if a_base is None:
            a_base = tuple((substitute(index, local | {k_axis}) for index in a.index))
            b_base = tuple((substitute(index, local | {k_axis}) for index in b.index))
            a_buffer, b_buffer = (a.input, b.input)
            if any((names_in(index) - (set(axis_names[:-2]) | {outer.axis.name}) for index in (*a_base, *b_base))):
                return None
        if a.input != a_buffer or b.input != b_buffer:
            return None
        expected_a = (add_terms(canonical(a_base[0]), (('var', row_axis), rows), ((), r)), add_terms(canonical(a_base[1]), (('var', k_axis), 1)))
        expected_b = (add_terms(canonical(b_base[0]), (('var', k_axis), 1)), add_terms(canonical(b_base[1]), (('var', col_axis), cols), ((), c)))
        if tuple(map(canonical, a.index)) != expected_a or tuple(map(canonical, b.index)) != expected_b:
            return None
    if any((statement.smem not in smem for statement in all_statements if kind(statement) == 'CpAsyncCopy')):
        return None
    return MatmulPlan(tile, inner, outer, a_buffer, b_buffer, a_base, b_base, output.output, base, output.atomic, tile_m, tile_n, tile_k, writes)
