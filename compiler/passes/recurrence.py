import math
from llvmlite import ir
from compiler.analysis import STATIC_SHARED_LIMIT, free_variables, index_variables, integer, kind, walk
from compiler.options import Recurrence
from compiler.transforms import SCALAR_OPS, sharing_width, symbolic, uniform_prefix_values
I1 = ir.IntType(1)
I32 = ir.IntType(32)
F32 = ir.FloatType()
VOID = ir.VoidType()

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
        self.loop_n(kaxis, p.reductions[0].axis.extent.value, score_step, unroll=32)
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

