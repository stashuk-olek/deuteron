import math
from typing import Any
from llvmlite import ir
from compiler.analysis import STATIC_SHARED_LIMIT, free_variables, index_variables, integer, kind, walk
from compiler.options import Recurrence
from compiler.transforms import SCALAR_OPS, sharing_width, symbolic, uniform_prefix_values
I1: Any = ir.IntType(1)
I8: Any = ir.IntType(8)
I32: Any = ir.IntType(32)
I64: Any = ir.IntType(64)
F32: Any = ir.FloatType()
VOID: Any = ir.VoidType()

def stage_recurrences(kernel: Any, launch: Any) -> tuple[Recurrence, ...]:
    readonly = {a['name'] for a in launch['args'] if not a.get('output')}
    found = []
    for tag, tile in enumerate(kernel.body):
        if kind(tile) != 'Tile':
            continue
        plan = _stage_tile(tile, launch, readonly, tag)
        if plan is not None:
            found.append(plan)
    return tuple(found)

def _stage_tile(tile: Any, launch: Any, readonly: Any, tag: Any) -> Any:
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

def _eval(expression: Any, env: Any) -> Any:
    if isinstance(expression, int):
        return expression
    tag = kind(expression)
    if tag == 'Literal':
        return expression.value if isinstance(expression.value, int) else None
    if tag == 'Var':
        return env.get(expression.name)
    if tag == 'BinaryExpr':
        left, right = (_eval(expression.left, env), _eval(expression.right, env))
        if left is None or right is None or not isinstance(left, int) or not isinstance(right, int):
            return None
        if expression.op == '+':
            return left + right
        if expression.op == '-':
            return left - right
        if expression.op == '*':
            return left * right
        if expression.op in ('/', '//'):
            return left // right if right else None
        if expression.op == '%':
            return left % right if right else None
        if expression.op == '<<':
            return left << right
        if expression.op == '>>':
            return left >> right
        if expression.op == '&':
            return left & right
        if expression.op == '|':
            return left | right
        if expression.op == '^':
            return left ^ right
    return None

def _cover_points(load: Any, shape: Any, ranges: Any) -> Any:
    if len(load.index) != len(shape):
        return None
    names = index_variables(load.index)
    if names is None or any((a not in ranges for a in names)):
        return None
    axes = sorted(names)
    total = 1
    for a in axes:
        total *= ranges[a]
        if total > 65536:
            return None
    n = math.prod(shape)
    seen = set()
    for ticket in range(total):
        env = {}
        for a in axes:
            env[a], ticket = (ticket % ranges[a], ticket // ranges[a])
        point = []
        for coord in load.index:
            value = _eval(coord, env)
            if value is None:
                return None
            point.append(value)
        flat = 0
        for value, extent in zip(point, shape):
            flat = flat * extent + value
        for k in range(len(load.names)):
            seen.add(flat + k)
        if len(seen) > n:
            return None
    return seen

def _full_cover(load: Any, shape: Any, ranges: Any) -> Any:
    seen = _cover_points(load, shape, ranges)
    if seen is None:
        return None
    n = math.prod(shape)
    return n * 4 if seen == set(range(n)) else None

def _stage_operands(tile: Any, outer: Any, reduction: Any, launch: Any, loads: Any, width: Any, extent: Any, parts: Any, used: Any) -> Any:
    if parts < 2:
        return ()
    if math.prod((a.extent.value for a in tile.axes)) != launch['grid'][0] * launch['block'][0]:
        return ()
    ranges = {a.name: a.extent.value for a in tile.axes}
    ranges[outer.axis.name] = extent
    ranges[reduction.axis.name] = reduction.axis.extent.value
    shapes = {a['name']: tuple(a['shape']) for a in launch['args']}
    rounds = extent // width
    block = launch['block'][0]
    by_buffer: dict[Any, Any] = {}
    for load in loads:
        names = index_variables(load.index)
        if names is None:
            continue
        shape = shapes.get(load.input)
        if not shape:
            continue
        seen = _cover_points(load, shape, ranges)
        if seen is None:
            continue
        if outer.axis.name in names:
            trips = reduction.axis.extent.value * rounds if reduction.axis.name in names else extent
        else:
            trips = reduction.axis.extent.value
        traffic = trips * block * len(load.names) * 4
        slot = by_buffer.setdefault(load.input, [set(), 0])
        slot[0].update(seen)
        slot[1] += traffic
    cands = []
    for name, (seen, traffic) in by_buffer.items():
        shape = shapes[name]
        nbytes = math.prod(shape) * 4
        if seen != set(range(math.prod(shape))) or nbytes % 16:
            continue
        if traffic < 2 * nbytes:
            continue
        cands.append((name, nbytes, traffic))
    cands.sort(key=lambda c: -c[2])
    staged, total, seen = ([], used, set())
    for name, nbytes, _ in cands:
        if name in seen or total + nbytes > STATIC_SHARED_LIMIT:
            continue
        seen.add(name)
        total += nbytes
        staged.append((name, nbytes))
    return tuple(staged)

def _recurrence(tile: Any, outer: Any, at: Any, launch: Any, readonly: Any, tag: Any) -> Any:
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

    def mass(op: Any) -> Any:
        return op.associative and op.identity == 0 and product_op.distributes_over(op)
    if any((not s.op.selecting and not mass(s.op) for s in states.values())):
        return None
    env = {n: ('state', n) for n in states}
    external = set()
    inputs = {}
    gather = []

    def get(n: Any) -> Any:
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
    staged = _stage_operands(tile, outer, reductions[0], launch, (varying[0], *invariant, *gather), width, extent, parts, count * 4)
    return Recurrence(tile, outer, prefix, suffix, reductions, tuple(invariant), tuple(varying), scale, width, extent, tuple(components), tuple(gather), next(iter(identities)), tag, staged)

class RecurrenceEmitter:

    def __init__(self, lowering: Any, plan: Recurrence) -> None:
        buffers_key = f'__staged_{plan.tag}'
        if buffers_key in lowering.buffers or f'staged_weights_{plan.tag}' in lowering.module.globals:
            raise ValueError('staged recurrence scratch already allocated')
        if any((name.startswith(f'__score_{plan.tag}_') or name.startswith(f'__normalizer_{plan.tag}_') for name in lowering.slots)):
            raise ValueError('staged recurrence slots already allocated')
        for index, (_name, _nbytes) in enumerate(plan.staged):
            if f'__operand_{plan.tag}_{index}' in lowering.buffers or f'staged_operand_{plan.tag}_{index}' in lowering.module.globals or f'__copy_{plan.tag}_{index}' in lowering.slots:
                raise ValueError('staged operand scratch already allocated')
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
        scratch: Any = ir.GlobalVariable(lowering.module, ir.ArrayType(F32, count), f'staged_weights_{plan.tag}', addrspace=3)
        scratch.linkage = 'internal'
        scratch.initializer = ir.Constant(scratch.value_type, ir.Undefined)
        scratch.align = 16
        lowering.buffers[self.staged] = lowering.builder.gep(scratch, [integer(0), integer(0)])
        lowering.shapes[self.staged] = (count,)
        self.operands = {}
        for index, (name, nbytes) in enumerate(plan.staged):
            key = f'__operand_{plan.tag}_{index}'
            scratch: Any = ir.GlobalVariable(lowering.module, ir.ArrayType(F32, nbytes // 4), f'staged_operand_{plan.tag}_{index}', addrspace=3)
            scratch.linkage = 'internal'
            scratch.initializer = ir.Constant(scratch.value_type, ir.Undefined)
            scratch.align = 16
            lowering.buffers[key] = lowering.builder.gep(scratch, [integer(0), integer(0)])
            lowering.shapes[key] = lowering.shapes[name]
            lowering.allocate(f'__copy_{plan.tag}_{index}', I32)
            self.operands[name] = key

    def stage_copy(self, source: Any, scratch: Any, nbytes: Any, index: Any) -> None:
        lowering = self.lowering
        name = f'__copy_{self.plan.tag}_{index}'
        block = lowering.launch['block'][0]
        builder = lowering.builder
        lowering.put(name, lowering.tid)
        header = lowering.function.append_basic_block(name + '.test')
        body = lowering.function.append_basic_block(name + '.body')
        end = lowering.function.append_basic_block(name + '.end')
        builder.branch(header)
        builder.position_at_end(header)
        builder.cbranch(builder.icmp_unsigned('<', lowering.get(name), integer(nbytes // 16)), body, end)
        builder.position_at_end(body)
        ticket = lowering.get(name)
        words = builder.mul(ticket, integer(4))
        dest = builder.ptrtoint(builder.gep(lowering.buffers[scratch], [words]), I32)
        src = builder.ptrtoint(builder.gep(lowering.buffers[source], [words]), I64)
        lowering.assembly('cp.async.cg.shared.global [$0], [$1], 16;', 'r,l,~{memory}', args=(dest, src))
        lowering.put(name, builder.add(ticket, integer(block)))
        builder.branch(header)
        builder.position_at_end(end)

    def stage_fence(self) -> None:
        lowering = self.lowering
        lowering.assembly('cp.async.commit_group;', '~{memory}')
        lowering.assembly('cp.async.wait_group 0;', '~{memory}')
        lowering.synchronize()

    def staged_load(self, load: Any) -> None:
        lowering = self.lowering
        scratch = self.operands[load.input]
        count = len(load.names)
        pointer = lowering.address(scratch, load.index)
        if count == 1:
            lowering.put(load.names[0], lowering.load(pointer, scratch, 4))
        elif count in (2, 4):
            vector = ir.VectorType(F32, count)
            pointer = lowering.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
            value = lowering.load(pointer, scratch, 4 * count)
            for i, name in enumerate(load.names):
                lowering.put(name, lowering.builder.extract_element(value, integer(i)))
        else:
            raise NotImplementedError(f'load width {count}')

    def loop_n(self, name: Any, extent: Any, callback: Any, unroll: Any = 8) -> None:
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
        md: Any = ir.MDValue(lowering.module, (), name=str(len(lowering.module.metadata)))
        hints = [hint]
        if name == self.plan.loop.axis.name:
            hints.extend([lowering.module.add_metadata(['llvm.loop.vectorize.enable', ir.Constant(I1, 1)]), lowering.module.add_metadata(['llvm.loop.vectorize.width', integer(8)]), lowering.module.add_metadata(['llvm.loop.interleave.count', integer(1)])])
        md.operands = (md, *hints)
        latch.set_metadata('llvm.loop', md)
        lowering.builder.position_at_end(end)

    def reduce(self, value: Any, width: Any, op: Any) -> Any:
        lowering = self.lowering
        distance = width // 2
        while distance:
            bits = lowering.builder.bitcast(value, I32)
            other = lowering.intrinsic('llvm.nvvm.shfl.sync.bfly.i32', I32, [integer(-1), bits, integer(distance), integer(31)])
            value = lowering.arithmetic(op, [value, lowering.builder.bitcast(other, F32)])
            distance //= 2
        return value

    def emit(self) -> None:
        lowering = self.lowering
        p = self.plan
        lowering.body(p.prefix)
        for index, (name, nbytes) in enumerate(p.staged):
            self.stage_copy(name, self.operands[name], nbytes, index)
        if p.staged:
            self.stage_fence()
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

        def score_step() -> None:
            for load in p.invariant:
                if load.input in self.operands:
                    self.staged_load(load)
                else:
                    lowering.body((load,))
            for r in range(rounds):
                lowering.put(keyaxis, lowering.builder.add(lane, integer(r * p.width)))
                varying = p.varying[0]
                if varying.input in self.operands:
                    self.staged_load(varying)
                else:
                    lowering.body((varying,))
                kval = lowering.get(p.varying[0].names[0])
                for part, load in enumerate(p.invariant):
                    name = f'{self.scores}_{part}_{r}'
                    product = lowering.arithmetic('multiply', [lowering.get(load.names[0]), kval])
                    lowering.put(name, lowering.arithmetic('add', [lowering.get(name), product]))
        self.loop_n(kaxis, p.reductions[0].axis.extent.value, score_step, unroll=64)
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

        def value_step() -> None:
            for load in p.gather:
                if load.input in self.operands:
                    self.staged_load(load)
                else:
                    lowering.body((load,))
            weights = []
            for part in range(parts):
                offset = lowering.builder.add(base, lowering.builder.add(lowering.get(keyaxis), integer(part * p.extent)))
                weights.append(lowering.builder.load(lowering.address(self.staged, (offset,)), align=4))
            for part, _m, _n, y, name_in in p.components:
                product = lowering.arithmetic('multiply', [weights[part], lowering.get(name_in)])
                lowering.put(y, lowering.builder.fadd(lowering.get(y), product, flags=('reassoc', 'nsz', 'contract')))
        self.loop_n(keyaxis, p.extent, value_step, unroll=64)
        for part, _m, n, _y, _name_in in p.components:
            lowering.put(n, lowering.get(f'{self.normalizer}_{part}'))
        lowering.body(p.suffix)
        lowering.striped = True

