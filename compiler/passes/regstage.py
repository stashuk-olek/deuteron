from dataclasses import replace
import itertools
import math
from typing import Any
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Load, Write
from compiler.analysis import canonical, exprs, index_variables, kind, known_multiple, walk
from compiler.passes.contraction import tensorize

def _rewrite(expr: Any, visit: Any) -> Any:
    found = visit(expr)
    if found is not None:
        return found
    if kind(expr) == 'BinaryExpr':
        return replace(expr, left=_rewrite(expr.left, visit), right=_rewrite(expr.right, visit))
    if kind(expr) == 'TernaryExpr':
        return replace(expr, cond=_rewrite(expr.cond, visit), if_true=_rewrite(expr.if_true, visit), if_false=_rewrite(expr.if_false, visit))
    return expr

def _evaluate(expr: Any, values: Any) -> int:
    if isinstance(expr, int):
        value = expr
    elif kind(expr) == 'Var':
        value = values[expr.name]
    elif kind(expr) == 'Literal' and expr.dtype == 'int':
        value = expr.value
    elif kind(expr) == 'BinaryExpr':
        a, b = _evaluate(expr.left, values), _evaluate(expr.right, values)
        value = {'+': lambda: a + b, '-': lambda: a - b, '*': lambda: a * b, '/': lambda: a // b, '//': lambda: a // b, '%': lambda: a % b}[expr.op]()
    else:
        raise ValueError('noninteger copy coordinate')
    if type(value) is not int or not 0 <= value < 2**31:
        raise ValueError('copy coordinate outside nonnegative i32')
    return value

def _integer_form(expr: Any) -> bool:
    if isinstance(expr, int):
        return type(expr) is int and 0 <= expr < 2**31
    if kind(expr) == 'Literal':
        return expr.dtype == 'int' and type(expr.value) is int and 0 <= expr.value < 2**31
    if kind(expr) == 'Var':
        return True
    if kind(expr) == 'TernaryExpr':
        return kind(expr.cond) == 'BinaryExpr' and expr.cond.op == '<' and all(_integer_form(v) for v in (expr.cond, expr.if_true, expr.if_false))
    if kind(expr) != 'BinaryExpr' or expr.op not in ('+', '-', '*', '/', '//', '%', '<'):
        return False
    if expr.op in ('/', '//', '%') and (kind(expr.right) != 'Literal' or type(expr.right.value) is not int or expr.right.value <= 0):
        return False
    return _integer_form(expr.left) and _integer_form(expr.right)

def _origin(expr: Any, axes: Any) -> Any:
    lower = upper = 0
    for term, coefficient in canonical(expr).items():
        if not term:
            lower += coefficient
            upper += coefficient
        elif len(term) == 2 and term[0] == 'var' and term[1] in axes:
            delta = coefficient * (axes[term[1]] - 1)
            lower += min(0, delta)
            upper += max(0, delta)
        else:
            raise ValueError('non-affine output origin')
    if not 0 <= lower <= upper < 2**31:
        raise ValueError('output origin outside nonnegative i32')
    return lower, upper

def stage_register_loads(kernel: Any, launch: Any) -> Any:
    try:
        return _stage(kernel, launch)
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None

def _stage(kernel: Any, launch: Any) -> Any:
    threads, grid = launch['block'][0], launch['grid'][0]
    if type(threads) is not int or type(grid) is not int or not 0 < threads <= 1024 or not 0 < threads * grid < 2**31:
        return None
    plans = tensorize(kernel, launch, 'bf16x3')
    if len(plans) != 1 or len(kernel.body) != 1:
        return None
    p = plans[0]
    tile, outer = p.tile, p.outer
    depth, extent = p.depth, outer.axis.extent.value
    counter = outer.axis.name
    if any(type(a.extent.value) is not int or a.extent.value <= 0 for a in tile.axes) or type(extent) is not int or canonical(outer.start) or canonical(outer.step) != {(): depth} or outer.end is not None or extent % depth or extent <= depth:
        return None
    if p.atomic or len(outer.body) < 6 or tuple(kind(s) for s in outer.body[-5:]) != ('CpAsyncCommit', 'CpAsyncWait', 'Sync', 'Loop', 'Sync') or outer.body[-2] is not p.loop:
        return None
    copies = outer.body[:-5]
    if not copies or any(kind(s) != 'CpAsyncCopy' or s.nbytes != 16 or s.swizzle != 'NONE' for s in copies):
        return None
    ahead = outer.body[-4].group
    stages = ahead + 1
    if type(ahead) is not int or not 0 < ahead < 8 or ahead * depth > extent or extent - depth + ahead * depth >= 2**31:
        return None
    prefix = tile.body[:tile.body.index(outer)]
    suffix = tile.body[tile.body.index(outer) + 1:]
    if tuple(suffix) != p.writes or any(kind(s) not in ('Smem', 'Init', 'CpAsyncCopy', 'CpAsyncCommit') for s in prefix):
        return None
    smem = {s.name: s for s in prefix if kind(s) == 'Smem'}
    shapes = {p.lhs_buffer: (p.rows, depth), p.rhs_buffer: (depth, p.cols)}
    cells = sum(rows * cols for rows, cols in shapes.values())
    if cells > 32768 or len(copies) * threads * 4 != cells or sum(kind(s) == 'Smem' for s in prefix) != 2 or set(smem) != set(shapes) or any(tuple(smem[b].extents) != (rows * stages, cols) or type(smem[b].align) is not int or smem[b].align < 16 or smem[b].align & (smem[b].align - 1) for b, (rows, cols) in shapes.items()):
        return None
    controls = {a.name for a in tile.axes} | {counter, p.loop.axis.name, 'lane', 'warp'}
    scalars = {name for s in walk(tile.body) if kind(s) in ('Load', 'Assign', 'Accum', 'Init') for name in (s.names if kind(s) == 'Load' else (s.name,))}
    if controls & scalars or any(name.startswith(('threadIdx.', 'blockIdx.', 'blockDim.', 'gridDim.')) for name in controls | scalars):
        return None
    products = {s.name: s.args for s in p.loop.body if kind(s) == 'Assign'}
    accumulated = {s.name for s in p.loop.body if kind(s) == 'Accum'}
    consumed = {name for s in p.loop.body if kind(s) == 'Accum' for name in products[s.value]}
    if consumed != {name for s in p.loop.body if kind(s) == 'Load' for name in s.names} or {s.name for s in prefix if kind(s) == 'Init'} != accumulated:
        return None
    if any(s.op.name != ('multiply' if kind(s) == 'Assign' else 'add') for s in p.loop.body if kind(s) in ('Assign', 'Accum')):
        return None
    if any(not _integer_form(e) for s in walk(tile.body) for e in exprs(s)):
        return None
    if any(canonical(v) for v in (p.lhs_base[1], p.rhs_base[1])):
        return None
    ring = Var(counter) / depth % stages
    if canonical(p.lhs_base[0]) != canonical(ring * p.rows) or canonical(p.rhs_base[0]) != canonical(ring * depth):
        return None
    args = {a['name']: a for a in launch['args']}
    output_shape = args[p.output_buffer]['shape']
    if set(smem) & set(args) or len(args) != len(launch['args']) or any(type(n) is not int or n <= 0 for n in output_shape) or math.prod(output_shape) >= 2**31:
        return None
    origins = [_origin(v, {a.name: a.extent.value for a in tile.axes[:-2]}) for v in p.output_base]
    if any(origin[1] + size > bound for origin, size, bound in zip(origins, (p.rows, p.cols), output_shape, strict=True)):
        return None
    current = Var(counter)
    future = current + ahead * depth
    future = TernaryExpr(cond=BinaryExpr(op='<', left=future, right=Literal(extent, dtype='int')), if_true=future, if_false=Literal(extent - depth, dtype='int'))

    def replace_future(expr: Any) -> Any:
        if kind(expr) != 'TernaryExpr':
            return None
        if canonical(expr.if_true) != canonical(current + ahead * depth) or canonical(expr.if_false) != {(): extent - depth} or kind(expr.cond) != 'BinaryExpr' or expr.cond.op != '<' or canonical(expr.cond.left) != canonical(expr.if_true) or canonical(expr.cond.right) != {(): extent}:
            raise ValueError('unrecognized copy prefetch')
        return current

    def drop_ring(expr: Any) -> Any:
        if kind(expr) == 'BinaryExpr' and expr.op == '%' and counter in (index_variables((expr,)) or set()):
            return Literal(0, dtype='int')
        return None

    records = []
    for copy in copies:
        if copy.smem not in shapes or copy.src not in args:
            return None
        source = args[copy.src]
        role = 1 if copy.smem == p.lhs_buffer else 0
        if source.get('output') or source.get('scalar') or source['dtype'] != 'f32' or len(source['shape']) != 2 or any(type(n) is not int or n <= 0 for n in source['shape']) or math.prod(source['shape']) >= 2**31 or source['shape'][role] != extent or source['shape'][-1] % 4:
            return None
        if source['shape'][1 - role] != args[p.output_buffer]['shape'][1 - role]:
            return None
        destination = tuple(a + b for a, b in zip(copy.smem_index, copy.lane_index or (Literal(0), Literal(0)), strict=True))
        local = tuple(_rewrite(v, drop_ring) for v in destination)
        rows, cols = shapes[copy.smem]
        if type(copy.lane_rows) is not int or copy.lane_rows < 0 or (copy.lane_rows and (copy.lane_index is None or copy.lane_rows * cols != threads * 4)):
            return None
        copy_ring = (current / depth + ahead) % stages * rows
        if canonical(destination[0]) != canonical(copy_ring + local[0]) or canonical(destination[1]) != canonical(local[1]):
            return None
        expected_future = (p.output_base[0] + local[0], future + local[1]) if role == 1 else (future + local[0], p.output_base[1] + local[1])
        if tuple(map(canonical, copy.src_index)) != tuple(map(canonical, expected_future)):
            return None
        src = tuple(_rewrite(v, replace_future) for v in copy.src_index)
        expected = (p.output_base[0] + local[0], current + local[1]) if role == 1 else (current + local[0], p.output_base[1] + local[1])
        if tuple(map(canonical, src)) != tuple(map(canonical, expected)) or known_multiple(_rewrite(src[-1], lambda e: Literal(0, dtype='int') if kind(e) == 'Var' and e.name == counter else None)) % 4:
            return None
        if (index_variables(local) or set()) - {a.name for a in tile.axes[-2:]}:
            return None
        records.append((copy, local, src))
    for buffer, (rows, cols) in shapes.items():
        touched = set()
        for values in itertools.product(*(range(a.extent.value) for a in tile.axes[-2:])):
            variables = dict(zip((a.name for a in tile.axes[-2:]), values, strict=True))
            for copy, local, src in records:
                if copy.smem != buffer:
                    continue
                row, col = (_evaluate(v, variables) for v in local)
                if col % 4:
                    return None
                for offset in range(4):
                    point = row, col + offset
                    if point in touched or not 0 <= row < rows or not 0 <= col + offset < cols:
                        return None
                    touched.add(point)
        if len(touched) != rows * cols:
            return None
    groups = [[]]
    for statement in prefix:
        if kind(statement) == 'CpAsyncCopy':
            groups[-1].append(statement)
        elif kind(statement) == 'CpAsyncCommit':
            groups.append([])
    if groups[-1] or len(groups) != ahead + 1:
        return None
    for stage, group in enumerate(groups[:-1]):
        if len(group) != len(records):
            return None
        for old, (copy, local, src) in zip(group, records, strict=True):
            expected_src = tuple(_rewrite(v, lambda e: Literal(stage * depth, dtype='int') if kind(e) == 'Var' and e.name == counter else None) for v in src)
            expected_destination = (local[0] + stage * shapes[copy.smem][0], local[1])
            actual_destination = tuple(a + b for a, b in zip(old.smem_index, old.lane_index or (Literal(0), Literal(0)), strict=True))
            if old.smem != copy.smem or old.src != copy.src or old.nbytes != 16 or old.swizzle != 'NONE' or old.lane_rows != copy.lane_rows or tuple(map(canonical, old.src_index)) != tuple(map(canonical, expected_src)) or tuple(map(canonical, actual_destination)) != tuple(map(canonical, expected_destination)):
                return None
    names = {a['name'] for a in launch['args']}
    for s in walk(tile.body):
        names.update(getattr(s, 'names', ()))
        if hasattr(s, 'name'):
            names.add(s.name)
        if hasattr(s, 'axis'):
            names.add(s.axis.name)
    names.update(a.name for a in tile.axes)
    temporary = '__operand'
    while any(name.startswith(temporary) for name in names):
        temporary += '_'
    initial, stores, loads = [], [], []
    next_value = current + depth
    next_value = TernaryExpr(cond=BinaryExpr(op='<', left=next_value, right=Literal(extent, dtype='int')), if_true=next_value, if_false=Literal(extent - depth, dtype='int'))
    for i, (copy, local, src) in enumerate(records):
        values = tuple(f'{temporary}_{i}_{j}' for j in range(4))
        load = Load(names=values, input=copy.src, index=src, dtype=F32)
        store = Write(output=copy.smem, index=local, values=values, value_dtype=F32)
        first = tuple(_rewrite(v, lambda e: Literal(0, dtype='int') if kind(e) == 'Var' and e.name == counter else None) for v in src)
        next_src = tuple(_rewrite(v, lambda e: next_value if kind(e) == 'Var' and e.name == counter else None) for v in src)
        initial.append(replace(load, index=first))
        stores.append(store)
        loads.append(replace(load, index=next_src))
    inner = replace(p.loop, body=Body(tuple(replace(s, index=tuple(_rewrite(v, drop_ring) for v in s.index)) if kind(s) == 'Load' else s for s in p.loop.body)))
    pending = BinaryExpr(op='<', left=current + depth, right=Literal(extent, dtype='int'))
    drain = Cond(cond=pending, body=Body((outer.body[-1],)))
    prefetch = Cond(cond=pending, body=Body(tuple(loads)))
    body = (*stores, outer.body[-3], prefetch, inner, drain)
    outer = replace(outer, body=Body(body))
    prefix = tuple(s for s in prefix if kind(s) in ('Smem', 'Init'))
    new_tile = replace(tile, body=Body((*prefix, *initial, outer, *suffix)))
    result = replace(kernel, body=Body((new_tile,)))
    plans = tensorize(result, launch, 'bf16x3')
    if len(plans) != 1 or any(canonical(v) for v in (*plans[0].lhs_base, *plans[0].rhs_base)) or any(getattr(plans[0], name) != getattr(p, name) for name in ('lhs_buffer', 'rhs_buffer', 'rows', 'cols', 'depth')):
        return None
    return result, (p.lhs_buffer, p.rhs_buffer)
