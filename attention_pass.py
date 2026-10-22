import math
from dataclasses import dataclass
from llvmlite import ir
from optimizations import uniform_dot_width
I1 = ir.IntType(1)
I32 = ir.IntType(32)
F32 = ir.FloatType()
VOID = ir.VoidType()
STATIC_SHARED_LIMIT = 48 * 1024
SCALAR_OPS = {'add', 'subtract', 'multiply', 'divide', 'copy', 'reciprocal', 'exp', 'sqrt', 'maximum', 'relu'}

def integer(value):
    return ir.Constant(I32, int(value))

def walk(body):
    for statement in body:
        yield statement
        for attr in ('body', 'else_body', 'combine_states'):
            yield from walk(getattr(statement, attr, ()))

def expr(op, *args):
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

def leaf(e, kind):
    return isinstance(e, tuple) and len(e) == 2 and (e[0] == kind)

def dependencies(expression):
    kind = type(expression).__name__
    if isinstance(expression, int) or kind == 'Literal':
        return set()
    if kind == 'Var':
        return {expression.name}
    if kind == 'BinaryExpr':
        if expression.op not in {'+', '-', '*', '/', '//', '%', '&', '|', '^', '<<', '>>', '==', '!=', '<', '<=', '>', '>=', '&&', '||'}:
            return None
        children = (expression.left, expression.right)
    elif kind == 'TernaryExpr':
        children = (expression.cond, expression.if_true, expression.if_false)
    else:
        return None
    return index_dependencies(children)

def index_dependencies(indices):
    result = set()
    for index in indices:
        names = dependencies(index)
        if names is None:
            return None
        result.update(names)
    return result

def uniform_prefix_values(prefix, tile, width):
    uniform_axes = {'warp'}
    for index, axis in enumerate(tile.axes):
        stride = math.prod((a.extent.value for a in tile.axes[index + 1:]))
        if axis.extent.value == 1 or stride % width == 0:
            uniform_axes.add(axis.name)
    values = {}
    for statement in prefix:
        kind = type(statement).__name__
        if kind == 'Load':
            names = index_dependencies(statement.index)
            if names is None:
                return None
            uniform = not names - uniform_axes
            values.update(((name, uniform) for name in statement.names))
        elif kind == 'Let':
            names = dependencies(statement.value)
            if names is None:
                return None
            known_uniform = uniform_axes | {name for name, uniform in values.items() if uniform}
            values[statement.name] = not names - known_uniform
        elif kind == 'Assign':
            if statement.op.name not in SCALAR_OPS:
                return None
            values[statement.name] = all((values.get(name, False) for name in statement.args))
        else:
            return None
    return values

@dataclass
class Plan:
    tile: object
    outer: object
    prefix: tuple
    suffix: tuple
    dots: tuple
    qloads: tuple
    ktloads: tuple
    scale: str
    width: int
    keys: int
    components: tuple
    vloads: tuple
    max_identity: float

def recognize_attention(tile):
    if not tile.block_threads or tile.block_threads % 32 or math.prod((a.extent.value for a in tile.axes)) % 32:
        return None
    loops = [(i, s) for i, s in enumerate(tile.body) if type(s).__name__ == 'Loop']
    if len(loops) != 1:
        return None
    at, outer = loops[0]
    if not outer.seed:
        return None
    prefix, suffix = (tile.body[:at], tile.body[at + 1:])
    if any((type(s).__name__ not in ('Load', 'Let', 'Assign') for s in prefix)):
        return None
    dots = tuple((s for s in outer.body if type(s).__name__ == 'Loop'))
    if not dots:
        return None
    widths = [uniform_dot_width(d, tile, {outer.axis.name}) for d in dots]
    if min(widths) <= 1 or min(widths) != max(widths):
        return None
    width = widths[0]
    keys = outer.axis.extent.value
    if keys <= 0 or keys % width or keys // width > 8:
        return None
    if len({(d.axis.name, d.axis.extent.value) for d in dots}) != 1:
        return None
    if dots[0].axis.name == outer.axis.name:
        return None
    dot_names = set()
    for dot in dots:
        left, right, product, accumulator = dot.body
        names = {left.names[0], right.names[0], product.name, accumulator.name}
        if len(names) != 4 or names & dot_names or accumulator.op.identity != 0:
            return None
        dot_names.update(names)
    qloads = []
    ktloads = []
    for dot in dots:
        q, k = dot.body[:2]
        q_names, k_names = (index_dependencies(q.index), index_dependencies(k.index))
        if q_names is None or k_names is None:
            return None
        if outer.axis.name in q_names:
            q, k = (k, q)
            q_names, k_names = (k_names, q_names)
        if outer.axis.name in q_names:
            return None
        if outer.axis.name not in k_names:
            return None
        qloads.append(q)
        ktloads.append(k)
    if any(((k.input, k.index) != (ktloads[0].input, ktloads[0].index) for k in ktloads)):
        return None
    qnames = [load.names[0] for load in qloads]
    knames = {load.names[0] for load in ktloads}
    if len(set(qnames)) != len(qnames) or set(qnames) & knames:
        return None
    states = {s.name: s for s in outer.body if type(s).__name__ == 'Accum'}
    if dot_names & set(states):
        return None
    if len(states) != sum((type(s).__name__ == 'Accum' for s in outer.body)):
        return None
    if any((s.op.name not in ('maximum', 'add') for s in states.values())):
        return None
    env = {n: ('state', n) for n in states}
    external = set()
    inputs = {}
    vloads = []

    def get(n):
        if n not in env:
            external.add(n)
            env[n] = ('external', n)
        return env[n]
    for s in outer.body:
        kind = type(s).__name__
        if kind == 'Loop':
            env[s.body[-1].name] = ('dot', dots.index(s))
        elif kind == 'Load':
            vloads.append(s)
            for n in s.names:
                env[n] = ('input', n)
                inputs[n] = s
        elif kind == 'Let':
            if type(s.value).__name__ != 'Literal':
                return None
            env[s.name] = ('const', float(s.value.value))
        elif kind == 'Assign':
            if s.op.name not in SCALAR_OPS:
                return None
            env[s.name] = expr(s.op.name, *(get(a) for a in s.args))
        elif kind == 'Accum':
            env[s.name] = expr(s.op.name, get(s.base or s.name), get(s.value))
        else:
            return None
    vnames = [name for load in vloads for name in load.names]
    if len(set(vnames)) != len(vnames) or set(vnames) & set(states):
        return None
    components = []
    covered = set()
    scales = set()
    maxidentities = set()
    for m, stmt in states.items():
        if stmt.op.name != 'maximum':
            continue
        expression = env[m]
        if expression[0] != 'maximum' or ('state', m) not in expression[1:]:
            return None
        scores = [e for e in expression[1:] if e != ('state', m)]
        if len(expression) != 3 or len(scores) != 1:
            return None
        score = scores[0]
        if score[0] != 'multiply' or len(score) != 3:
            return None
        dot = next((a for a in score[1:] if leaf(a, 'dot')), None)
        scale = next((a for a in score[1:] if leaf(a, 'external')), None)
        if dot is None or scale is None:
            return None
        scales.add(scale[1])
        row = dot[1]
        maxidentities.add(stmt.op.identity)
        alpha = expr('exp', expr('subtract', ('state', m), expression))
        beta = expr('exp', expr('subtract', score, expression))
        norms = []
        weights = []
        for name, state in states.items():
            if state.op.name != 'add' or state.op.identity != 0:
                continue
            carried = expr('multiply', ('state', name), alpha)
            if env[name] == expr('add', carried, beta):
                norms.append(name)
            for inp in inputs:
                if env[name] == expr('add', carried, expr('multiply', ('input', inp), beta)):
                    weights.append((name, inp))
        if len(norms) != 1 or len(weights) != 1:
            return None
        components.append((row, m, norms[0], weights[0][0], weights[0][1]))
        covered.update((m, norms[0], weights[0][0]))
    if covered != set(states) or len(scales) != 1 or len(maxidentities) != 1:
        return None
    if math.isnan(next(iter(maxidentities))):
        return None
    definitions = set()
    for statement in walk(outer.body):
        kind = type(statement).__name__
        if kind == 'Load':
            definitions.update(statement.names)
        elif kind in ('Assign', 'Accum', 'Let', 'Init'):
            definitions.add(statement.name)
    if external & definitions:
        return None
    scale = next(iter(scales))
    prefix_values = uniform_prefix_values(prefix, tile, width)
    if prefix_values is None or not prefix_values.get(scale, False):
        return None
    preserved_coordinates = {axis.name for axis in tile.axes} | {'lane', 'warp'}
    counter_names = preserved_coordinates | {outer.axis.name, dots[0].axis.name}
    if definitions & counter_names or set(prefix_values) & counter_names:
        return None
    final = {name: ('final', name) for name in states}
    written = set()
    expected = {expr('multiply', ('final', y), expr('reciprocal', ('final', n))): (row, y) for row, m, n, y, inp in components}
    for s in suffix:
        if type(s).__name__ == 'Assign':
            if s.op.name not in SCALAR_OPS or s.name in counter_names or any((a not in final for a in s.args)):
                return None
            final[s.name] = expr(s.op.name, *(final[a] for a in s.args))
        elif type(s).__name__ == 'Write':
            if s.atomic or s.swizzle != 'NONE':
                return None
            coordinates = index_dependencies(s.index)
            if coordinates is None or coordinates - preserved_coordinates:
                return None
            for name in s.values:
                if final.get(name) not in expected:
                    return None
                written.add(expected[final[name]][1])
        else:
            return None
    if written != {c[3] for c in components}:
        return None
    if {c[0] for c in components} != set(range(len(dots))):
        return None
    allowed = {a.name for a in tile.axes} | {outer.axis.name}
    for load in vloads:
        coordinates = index_dependencies(load.index)
        if coordinates is None or coordinates - allowed:
            return None
    return Plan(tile, outer, prefix, suffix, dots, tuple(qloads), tuple(ktloads), scale, width, keys, tuple(components), tuple(vloads), next(iter(maxidentities)))

class AttentionEmitter:

    def __init__(self, lowering, plan):
        self.lowering = lowering
        self.plan = plan
        rows = len(plan.dots)
        rounds = plan.keys // plan.width
        for row in range(rows):
            for key in range(rounds):
                lowering.allocate(f'__score_{row}_{key}', F32)
            lowering.allocate(f'__normalizer_{row}', F32)
        count = lowering.launch['block'][0] // plan.width * rows * plan.keys
        scratch = ir.GlobalVariable(lowering.module, ir.ArrayType(F32, count), 'attention_scores', addrspace=3)
        scratch.linkage = 'internal'
        scratch.initializer = ir.Constant(scratch.value_type, ir.Undefined)
        scratch.align = 16
        lowering.buffers['__scores'] = lowering.builder.gep(scratch, [integer(0), integer(0)])
        lowering.shapes['__scores'] = (count,)

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
        if name == self.plan.outer.axis.name:
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
        rows = len(p.dots)
        rounds = p.keys // p.width
        keyaxis = p.outer.axis.name
        kaxis = p.dots[0].axis.name
        lane = lowering.builder.urem(lowering.get('lane'), integer(p.width))
        group = lowering.builder.udiv(lowering.tid, integer(p.width))
        base = lowering.builder.mul(group, integer(rows * p.keys))
        scale = lowering.get(p.scale)
        for row in range(rows):
            for r in range(rounds):
                lowering.put(f'__score_{row}_{r}', ir.Constant(F32, 0.0))

        def score_step():
            for qload in p.qloads:
                lowering.body((qload,))
            for r in range(rounds):
                lowering.put(keyaxis, lowering.builder.add(lane, integer(r * p.width)))
                lowering.body((p.ktloads[0],))
                kval = lowering.get(p.ktloads[0].names[0])
                for row, qload in enumerate(p.qloads):
                    name = f'__score_{row}_{r}'
                    product = lowering.arithmetic('multiply', [lowering.get(qload.names[0]), kval])
                    lowering.put(name, lowering.arithmetic('add', [lowering.get(name), product]))
        self.loop_n(kaxis, p.dots[0].axis.extent.value, score_step, unroll=32 if len(p.dots) == 1 else 8)
        for row in range(rows):
            scores = [lowering.arithmetic('multiply', [lowering.get(f'__score_{row}_{r}'), scale]) for r in range(rounds)]
            maximum = ir.Constant(F32, p.max_identity)
            for score in scores:
                maximum = lowering.arithmetic('maximum', [maximum, score])
            maximum = self.reduce(maximum, p.width, 'maximum')
            mass = ir.Constant(F32, 0.0)
            for r, score in enumerate(scores):
                weight = lowering.arithmetic('exp', [lowering.arithmetic('subtract', [score, maximum])])
                offset = lowering.builder.add(base, lowering.builder.add(lane, integer(row * p.keys + r * p.width)))
                lowering.builder.store(weight, lowering.address('__scores', (offset,)), align=4)
                mass = lowering.arithmetic('add', [mass, weight])
            mass = self.reduce(mass, p.width, 'add')
            lowering.put(f'__normalizer_{row}', mass)
        lowering.intrinsic('llvm.nvvm.bar.warp.sync', VOID, [integer(-1)])
        for _row, _m, _n, y, _inp in p.components:
            lowering.put(y, ir.Constant(F32, 0.0))

        def value_step():
            lowering.body(p.vloads)
            weights = []
            for row in range(rows):
                offset = lowering.builder.add(base, lowering.builder.add(lowering.get(keyaxis), integer(row * p.keys)))
                weights.append(lowering.builder.load(lowering.address('__scores', (offset,)), align=4))
            for row, _m, _n, y, inp in p.components:
                product = lowering.arithmetic('multiply', [weights[row], lowering.get(inp)])
                lowering.put(y, lowering.builder.fadd(lowering.get(y), product, flags=('reassoc', 'nsz', 'contract')))
        self.loop_n(keyaxis, p.keys, value_step, unroll=max(1, 16 // len(p.dots)))
        for row, _m, n, _y, _inp in p.components:
            lowering.put(n, lowering.get(f'__normalizer_{row}'))
        lowering.body(p.suffix)
        lowering.uses_cooperative_dots = True

def prepare_attention(lowering):
    kernel = lowering.kernel
    if len(kernel.body) != 1 or type(kernel.body[0]).__name__ != 'Tile':
        return None
    plan = recognize_attention(kernel.body[0])
    if plan is None or lowering.launch['block'][0] != plan.tile.block_threads:
        return None
    prefix_loads = tuple((statement for statement in plan.prefix if type(statement).__name__ == 'Load'))
    if any((load.input not in lowering.readonly for load in (*prefix_loads, *plan.qloads, *plan.ktloads, *plan.vloads))):
        return None
    rows = len(plan.dots)
    count = lowering.launch['block'][0] // plan.width * rows * plan.keys
    if count * 4 > STATIC_SHARED_LIMIT or rows * (plan.keys // plan.width) > 32:
        return None
    if '__scores' in lowering.buffers or 'attention_scores' in lowering.module.globals:
        return None
    if any((name.startswith(('__score_', '__normalizer_')) for name in lowering.slots)):
        return None
    return AttentionEmitter(lowering, plan)
