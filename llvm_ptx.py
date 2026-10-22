import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from emmy.compiler.graph import Graph
from llvmlite import binding as llvm
from llvmlite import ir
from attention_pass import prepare_attention
from matmul_pass import recognize_matmul
from mma_emitter import MMAEmitter
from optimizations import Options, contiguous_sum, default_unroll, reciprocal_sqrt_source, simple_sum, uniform_dot_width
I1 = ir.IntType(1)
I8 = ir.IntType(8)
I32 = ir.IntType(32)
I64 = ir.IntType(64)
F32 = ir.FloatType()
VOID = ir.VoidType()
STATIC_SHARED_LIMIT = 48 * 1024

def integer(value):
    return ir.Constant(I32, int(value.value if hasattr(value, 'as_static') else value))

def walk(body):
    for stmt in body:
        yield stmt
        for attr in ('body', 'else_body', 'combine_states'):
            yield from walk(getattr(stmt, attr, ()))

@dataclass(frozen=True)
class Compilation:
    ptx: str
    unoptimized: str
    optimized: str

class Lowering:

    def __init__(self, kernel, launch, module, options):
        self.kernel = kernel
        self.launch = launch
        self.module = module
        self.options = options
        if options.dot_precision not in ('f32', 'tf32x3', 'bf16x3'):
            raise ValueError(f'unsupported dot precision {options.dot_precision}')
        self.mma_emitter = None
        self.tile_stmt = None
        self.uniform_axes = set()
        self.divergent_depth = 0
        self.reassociate = False
        self.uses_cooperative_dots = False
        self.slots = {}
        self.buffers = {}
        self.shapes = {}
        specs = {a['name']: a for a in launch['args']}
        self.readonly = {a['name'] for a in launch['args'] if not a.get('output')}
        if len(specs) != len(launch['args']) or set(specs) != set(launch['arg_order']):
            raise ValueError('meta.json args and arg_order disagree')
        for a in specs.values():
            if a['dtype'] != 'f32' or a.get('scalar'):
                raise NotImplementedError('the open-set backend accepts f32 buffer arguments')
        if kernel.serial:
            raise NotImplementedError('serial launch axes are outside the open set')
        if launch['block'][1:] != [1, 1] or launch['grid'][1:] != [1, 1]:
            raise NotImplementedError('only linear launches are supported')
        signature = ir.FunctionType(VOID, [F32.as_pointer(1)] * len(specs))
        self.function = ir.Function(module, signature, kernel.name)
        self.function.calling_convention = 'ptx_kernel'
        module.add_named_metadata('nvvm.annotations', [self.function, 'kernel', integer(1)])
        module.add_named_metadata('nvvm.annotations', [self.function, 'maxntidx', integer(launch['block'][0])])
        self.builder = ir.IRBuilder(self.function.append_basic_block('entry'))
        for arg, name in zip(self.function.args, launch['arg_order'], strict=True):
            arg.name = name
            arg.attributes.add('noalias')
            arg.attributes.align = 16
            arg.attributes.dereferenceable = math.prod(specs[name]['shape']) * 4
            self.buffers[name] = arg
            self.shapes[name] = tuple(specs[name]['shape'])
        for stmt in walk(kernel.body):
            kind = type(stmt).__name__
            if kind in ('Load', 'Assign', 'Accum', 'Init', 'Let'):
                dtype = getattr(stmt, 'dtype', None)
                if dtype is not None and dtype.name != 'f32':
                    raise NotImplementedError(f'{kind}: unsupported dtype {dtype.name}')
                names = stmt.names if kind == 'Load' else (stmt.name,)
                for name in names:
                    self.allocate(name, F32)
            if kind in ('WarpShuffle', 'TreeHalve'):
                for name in (*stmt.state, *stmt.state_b):
                    self.allocate(name, F32)
            if kind in ('Loop', 'StridedLoop'):
                self.allocate(stmt.axis.name, I32)
                if self.options.coalesced_reductions and simple_sum(stmt):
                    accum = next((s for s in stmt.body if type(s).__name__ == 'Accum'))
                    self.allocate('__vector_' + accum.name, ir.VectorType(F32, 4))
            if kind == 'Tile':
                for axis in stmt.axes:
                    self.allocate(axis.name, I32)
        for name in ('lane', 'warp'):
            self.allocate(name, I32)
        self.shared_memory()
        tid = self.intrinsic('llvm.nvvm.read.ptx.sreg.tid.x', I32, [])
        self.tid = tid
        self.put('lane', self.builder.and_(tid, integer(31)))
        self.put('warp', self.builder.lshr(tid, integer(5)))
        self.attention_emitter = prepare_attention(self) if self.options.staged_attention else None
        if options.dot_precision != 'f32':
            plan = recognize_matmul(kernel, launch)
            if plan is not None:
                instruction_k = 16 if options.dot_precision == 'bf16x3' else 8
                active_threads = plan.tile_m // 16 * (plan.tile_n // 8) * 32
                if plan.tile_k % instruction_k == 0 and active_threads == launch['block'][0]:
                    self.mma_emitter = MMAEmitter(self, plan, options.dot_precision)

    def allocate(self, name, dtype):
        if name not in self.slots:
            self.slots[name] = self.builder.alloca(dtype, name=name + '.slot')
        elif self.slots[name].type.pointee != dtype:
            raise ValueError(f'inconsistent type for {name}')

    def get(self, name):
        if name not in self.slots:
            raise ValueError(f'unbound value {name}')
        return self.builder.load(self.slots[name], name=name)

    def put(self, name, value):
        self.builder.store(value, self.slots[name])

    def intrinsic(self, name, result, args):
        signature = ir.FunctionType(result, [a.type for a in args])
        function = self.module.globals.get(name)
        if function is None:
            function = ir.Function(self.module, signature, name)
        call = self.builder.call(function, args)
        if self.options.warp_willreturn and name in ('llvm.nvvm.activemask', 'llvm.nvvm.shfl.sync.bfly.i32'):
            set.add(call.attributes, 'willreturn')
        return call

    def assembly(self, text, constraints, result=VOID, args=()):
        signature = ir.FunctionType(result, [a.type for a in args])
        asm = ir.InlineAsm(signature, text, constraints, side_effect=True)
        return self.builder.call(asm, args)

    def shared_memory(self):
        declarations = {}
        offsets = {}
        size = 0
        for stmt in walk(self.kernel.body):
            if type(stmt).__name__ != 'Smem':
                continue
            if stmt.name in declarations:
                raise ValueError(f'duplicate shared buffer {stmt.name}')
            if stmt.dtype != 'float':
                raise NotImplementedError(f'shared dtype {stmt.dtype}')
            alignment = max(4, stmt.align)
            size = (size + alignment - 1) // alignment * alignment
            offsets[stmt.name] = size
            size += math.prod(stmt.extents) * 4
            declarations[stmt.name] = stmt
        dynamic = size > STATIC_SHARED_LIMIT
        if dynamic:
            if size > self.launch['smem_bytes']:
                raise ValueError(f"launch provides {self.launch['smem_bytes']} shared bytes; IR needs {size}")
            pool = ir.GlobalVariable(self.module, ir.ArrayType(I8, 0), 'dynamic_shared', addrspace=3)
            pool.linkage = 'external'
            pool.align = max((s.align or 4 for s in declarations.values()))
        for name, stmt in declarations.items():
            self.shapes[name] = tuple(stmt.extents)
            if dynamic:
                pointer = self.builder.gep(pool, [integer(0), integer(offsets[name])])
                self.buffers[name] = self.builder.bitcast(pointer, F32.as_pointer(3))
            else:
                array = ir.GlobalVariable(self.module, ir.ArrayType(F32, math.prod(stmt.extents)), name, addrspace=3)
                array.linkage = 'internal'
                array.initializer = ir.Constant(array.value_type, ir.Undefined)
                array.align = max(4, stmt.align)
                self.buffers[name] = self.builder.gep(array, [integer(0), integer(0)])

    def expression(self, expr):
        if isinstance(expr, int):
            return integer(expr)
        kind = type(expr).__name__
        if kind == 'Var':
            return self.get(expr.name)
        if kind == 'Literal':
            return ir.Constant(F32, float(expr.value)) if expr.dtype == 'float' else integer(expr.value)
        if kind == 'TernaryExpr':
            return self.builder.select(self.expression(expr.cond), self.expression(expr.if_true), self.expression(expr.if_false))
        if kind == 'BinaryExpr':
            left, right = (self.expression(expr.left), self.expression(expr.right))
            op = expr.op
            if op in ('==', '!=', '<', '<=', '>', '>='):
                return self.builder.icmp_signed(op, left, right)
            method = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'sdiv', '//': 'sdiv', '%': 'srem', '&': 'and_', '|': 'or_', '^': 'xor', '<<': 'shl', '>>': 'ashr', '&&': 'and_', '||': 'or_'}.get(op)
            if method:
                return getattr(self.builder, method)(left, right)
        raise NotImplementedError(f'expression {expr!r}')

    def address(self, buffer, index, extra=0):
        if self.mma_emitter is not None:
            index = self.mma_emitter.swizzle_shared_index(buffer, index)
        shape = self.shapes[buffer]
        if not index and math.prod(shape) == 1:
            shape = ()
        if len(index) != len(shape):
            raise ValueError(f'{buffer}: index rank {len(index)} does not match shape {shape}')
        offset = integer(0)
        for coordinate, extent in zip(index, shape, strict=True):
            value = coordinate if isinstance(coordinate, ir.Value) else self.expression(coordinate)
            offset = self.builder.add(self.builder.mul(offset, integer(extent)), value)
        if extra:
            offset = self.builder.add(offset, integer(extra))
        return self.builder.gep(self.buffers[buffer], [self.builder.sext(offset, I64)])

    def load(self, pointer, buffer, alignment):
        value = self.builder.load(pointer, align=alignment)
        if buffer in self.readonly:
            value.set_metadata('invariant.load', self.module.add_metadata([]))
        return value

    def arithmetic(self, op, args):
        method = {'add': 'fadd', 'subtract': 'fsub', 'multiply': 'fmul', 'divide': 'fdiv'}.get(op)
        if method:
            return getattr(self.builder, method)(*args, flags=('contract',))
        if op == 'copy':
            return args[0]
        if op == 'reciprocal':
            if self.options.approximate_math:
                return self.intrinsic('llvm.nvvm.rcp.approx.ftz.f', F32, args)
            return self.builder.fdiv(ir.Constant(F32, 1.0), args[0])
        if op == 'exp':
            value = self.builder.fmul(args[0], ir.Constant(F32, math.log2(math.e)))
            return self.intrinsic('llvm.nvvm.ex2.approx.f', F32, [value])
        if op == 'sqrt':
            if self.options.approximate_math:
                return self.intrinsic('llvm.nvvm.sqrt.approx.ftz.f', F32, args)
            return self.intrinsic('llvm.sqrt.f32', F32, args)
        if op in ('maximum', 'relu'):
            if op == 'relu':
                args = [args[0], ir.Constant(F32, 0.0)]
            return self.intrinsic('llvm.maxnum.f32', F32, args)
        raise NotImplementedError(f'elementwise operation {op}')

    def loop(self, stmt):
        if stmt.seed:
            seeded = set()
            for child in stmt.body:
                if type(child).__name__ == 'Accum' and child.name not in seeded:
                    self.put(child.name, ir.Constant(F32, child.op.identity))
                    seeded.add(child.name)
        if type(stmt).__name__ == 'Loop':
            start, step, end = (integer(0), integer(1), integer(stmt.axis.extent))
        else:
            start, step = (self.expression(stmt.start), self.expression(stmt.step))
            end = self.expression(stmt.end) if stmt.end is not None else integer(stmt.axis.extent)
        cooperative = 1
        if self.options.cooperative_dots and self.tile_stmt is not None and (self.divergent_depth == 0) and (self.launch['block'][0] % 32 == 0):
            cooperative = uniform_dot_width(stmt, self.tile_stmt, self.uniform_axes)
        if cooperative > 1:
            self.uses_cooperative_dots = True
            start = self.builder.urem(self.get('lane'), integer(cooperative))
            step = integer(cooperative)
        coalesced = self.options.coalesced_reductions and self.tile_stmt is not None and contiguous_sum(stmt, self.tile_stmt, self.launch['block'][0])
        if coalesced:
            start = self.builder.mul(start, integer(4))
            step = self.builder.mul(step, integer(4))
        vectorize = self.options.vector_width > 1 and simple_sum(stmt) and (not coalesced)
        if coalesced:
            vector_name = '__vector_' + stmt.body[-1].name
            self.put(vector_name, ir.Constant(ir.VectorType(F32, 4), [0.0] * 4))
        self.put(stmt.axis.name, start)
        header = self.function.append_basic_block(stmt.axis.name + '.test')
        body = self.function.append_basic_block(stmt.axis.name + '.body')
        exit_block = self.function.append_basic_block(stmt.axis.name + '.end')
        self.builder.branch(header)
        self.builder.position_at_end(header)
        self.builder.cbranch(self.builder.icmp_signed('<', self.get(stmt.axis.name), end), body, exit_block)
        self.builder.position_at_end(body)
        old_reassociate = self.reassociate
        self.reassociate = vectorize
        uniform = type(stmt).__name__ == 'Loop' and cooperative == 1
        if uniform:
            self.uniform_axes.add(stmt.axis.name)
        else:
            self.divergent_depth += 1
        if coalesced:
            load = stmt.body[0]
            pointer = self.address(load.input, load.index)
            vector = ir.VectorType(F32, 4)
            pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
            values = self.load(pointer, load.input, 16)
            if len(stmt.body) == 3:
                values = self.builder.fmul(values, values, flags=('contract',))
            result = self.builder.fadd(self.get(vector_name), values, flags=('reassoc', 'nsz', 'contract'))
            self.put(vector_name, result)
        else:
            self.body(stmt.body)
        if uniform:
            self.uniform_axes.remove(stmt.axis.name)
        else:
            self.divergent_depth -= 1
        self.reassociate = old_reassociate
        self.put(stmt.axis.name, self.builder.add(self.get(stmt.axis.name), step))
        latch = self.builder.branch(header)
        count = self.options.unroll or default_unroll(stmt, self.tile_stmt, self.uniform_axes)
        hint = ['llvm.loop.unroll.count', integer(count)] if count > 1 else ['llvm.loop.unroll.disable']
        metadata = ir.MDValue(self.module, (), name=str(len(self.module.metadata)))
        hints = [self.module.add_metadata(hint)]
        if vectorize:
            hints += [self.module.add_metadata(['llvm.loop.vectorize.enable', ir.Constant(I1, 1)]), self.module.add_metadata(['llvm.loop.vectorize.width', integer(self.options.vector_width)]), self.module.add_metadata(['llvm.loop.interleave.count', integer(1)])]
        metadata.operands = (metadata, *hints)
        latch.set_metadata('llvm.loop', metadata)
        self.builder.position_at_end(exit_block)
        if coalesced:
            partial = self.intrinsic('llvm.vector.reduce.fadd.v4f32', F32, [ir.Constant(F32, 0.0), self.get(vector_name)])
            partial.fastmath.add('reassoc')
            partial.fastmath.add('nsz')
            self.put(stmt.body[-1].name, partial)
        if cooperative > 1:
            name = stmt.body[-1].name
            value = self.get(name)
            mask = integer(-1) if self.launch['block'][0] % 32 == 0 else self.intrinsic('llvm.nvvm.activemask', I32, [])
            distance = cooperative // 2
            while distance:
                bits = self.builder.bitcast(value, I32)
                other = self.intrinsic('llvm.nvvm.shfl.sync.bfly.i32', I32, [mask, bits, integer(distance), integer(31)])
                value = self.builder.fadd(value, self.builder.bitcast(other, F32))
                distance //= 2
            self.put(name, value)

    def tile(self, stmt):
        if stmt.aux_threads:
            raise NotImplementedError('auxiliary threads are outside the open set')
        if stmt.block_threads and stmt.block_threads != self.launch['block'][0]:
            raise ValueError('Tile block_threads disagrees with the launch')
        block = self.intrinsic('llvm.nvvm.read.ptx.sreg.ctaid.x', I32, [])
        threads = self.intrinsic('llvm.nvvm.read.ptx.sreg.ntid.x', I32, [])
        gid = self.builder.add(self.builder.mul(block, threads), self.tid)
        size = math.prod((a.extent.value for a in stmt.axes))
        with self.builder.if_then(self.builder.icmp_unsigned('<', gid, integer(size))):
            stride = size
            for axis in stmt.axes:
                extent = axis.extent.value
                stride //= extent
                value = self.builder.urem(self.builder.udiv(gid, integer(stride)), integer(extent))
                self.put(axis.name, value)
            if stmt.raster_group and stmt.raster_axes:
                names = [axis.name for axis in stmt.axes]
                mi, ni = (names.index(name) for name in stmt.raster_axes)
                if ni != mi + 1:
                    raise NotImplementedError('raster axes must be adjacent')
                gi, oi = (mi, ni) if stmt.raster_orient == 'm' else (ni, mi)
                ea, eb = (stmt.axes[gi].extent.value, stmt.axes[oi].extent.value)
                group = min(stmt.raster_group, ea)
                inner = math.prod((a.extent.value for a in stmt.axes[ni + 1:]))
                sub = self.builder.urem(self.builder.udiv(gid, integer(inner)), integer(ea * eb))
                base = self.builder.mul(self.builder.udiv(sub, integer(group * eb)), integer(group))
                remaining = self.builder.sub(integer(ea), base)
                width = self.builder.select(self.builder.icmp_unsigned('<', remaining, integer(group)), remaining, integer(group))
                within = self.builder.urem(sub, integer(group * eb))
                self.put(names[gi], self.builder.add(base, self.builder.urem(within, width)))
                self.put(names[oi], self.builder.udiv(within, width))
            old_tile = self.tile_stmt
            self.tile_stmt = stmt if size % 32 == 0 else None
            if self.attention_emitter is not None and stmt is self.attention_emitter.plan.tile:
                self.attention_emitter.emit()
            else:
                self.body(stmt.body)
            self.tile_stmt = old_tile

    def synchronize(self, barrier_id=0, count=None, warp=False):
        if warp:
            mask = self.intrinsic('llvm.nvvm.activemask', I32, [])
            self.intrinsic('llvm.nvvm.bar.warp.sync', VOID, [mask])
        elif barrier_id:
            if count is None:
                raise ValueError('named barrier requires a thread count')
            self.assembly(f'bar.sync {barrier_id}, {count};', '~{memory}')
        else:
            self.intrinsic('llvm.nvvm.barrier0', VOID, [])

    def shuffle(self, stmt):
        if stmt.length < 1 or stmt.length > 32 or stmt.length & stmt.length - 1:
            raise ValueError('shuffle length must be a power of two up to 32')
        mask = self.intrinsic('llvm.nvvm.activemask', I32, [])
        distance = stmt.length // 2
        while distance:
            for state, other in zip(stmt.state, stmt.state_b, strict=True):
                bits = self.builder.bitcast(self.get(state), I32)
                value = self.intrinsic('llvm.nvvm.shfl.sync.bfly.i32', I32, [mask, bits, integer(distance), integer(31)])
                self.put(other, self.builder.bitcast(value, F32))
            self.body(stmt.combine_states)
            distance //= 2

    def tree(self, stmt):
        if stmt.length < 1 or stmt.length & stmt.length - 1:
            raise ValueError('tree length must be a power of two')
        if self.options.shuffle_tree and stmt.tid_var == 'warp' and (stmt.inner is None) and (stmt.length <= 32):
            with self.builder.if_then(self.builder.icmp_unsigned('==', self.get('warp'), integer(0))):
                slot = self.builder.and_(self.get('lane'), integer(stmt.length - 1))
                for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                    self.put(state, self.builder.load(self.address(buf, (slot,))))
                self.shuffle(stmt)
                with self.builder.if_then(self.builder.icmp_unsigned('==', self.get('lane'), integer(0))):
                    for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                        self.builder.store(self.get(state), self.address(buf, (integer(0),)))
            self.synchronize(stmt.barrier_id, stmt.barrier_count)
            for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                self.put(state, self.builder.load(self.address(buf, (integer(0),))))
            return
        tid = self.get(stmt.tid_var)
        scale = stmt.inner[1] if stmt.inner else 1
        root = self.get(stmt.inner[0]) if stmt.inner else integer(0)
        slot = self.builder.add(self.builder.mul(tid, integer(scale)), root)
        distance = stmt.length // 2
        while distance:
            active = self.builder.icmp_unsigned('<', tid, integer(distance))
            if stmt.tid_var == 'warp':
                active = self.builder.and_(active, self.builder.icmp_unsigned('==', self.get('lane'), integer(0)))
            with self.builder.if_then(active):
                neighbor = self.builder.add(slot, integer(distance * scale))
                for buf, state, other in zip(stmt.bufs, stmt.state, stmt.state_b, strict=True):
                    self.put(state, self.builder.load(self.address(buf, (slot,))))
                    self.put(other, self.builder.load(self.address(buf, (neighbor,))))
                self.body(stmt.combine_states)
                for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                    self.builder.store(self.get(state), self.address(buf, (slot,)))
            self.synchronize(stmt.barrier_id, stmt.barrier_count)
            distance //= 2
        for buf, state in zip(stmt.bufs, stmt.state, strict=True):
            self.put(state, self.builder.load(self.address(buf, (root,))))

    def async_copy(self, stmt):
        if stmt.swizzle != 'NONE':
            raise NotImplementedError('shared-memory swizzling is outside the open set')
        if stmt.nbytes not in (4, 8, 16):
            raise ValueError('cp.async copy size must be 4, 8, or 16')
        index = [self.expression(x) for x in stmt.smem_index]
        if stmt.lane_index is not None:
            index = [self.builder.add(x, self.expression(y)) for x, y in zip(index, stmt.lane_index, strict=True)]
        destination = self.builder.ptrtoint(self.address(stmt.smem, index), I32)
        source = self.builder.ptrtoint(self.address(stmt.src, stmt.src_index), I64)
        self.assembly(f'cp.async.ca.shared.global [$0], [$1], {stmt.nbytes};', 'r,l,~{memory}', args=(destination, source))

    def body(self, body):
        for position, stmt in enumerate(body):
            kind = type(stmt).__name__
            if self.mma_emitter is not None:
                if stmt is self.mma_emitter.plan.inner_loop:
                    self.mma_emitter.emit_dot()
                    continue
                if kind == 'Write' and self.mma_emitter.emit_write(stmt):
                    continue
            if kind == 'Tile':
                self.tile(stmt)
            elif kind in ('Loop', 'StridedLoop'):
                self.loop(stmt)
            elif kind == 'Load':
                count = len(stmt.names)
                pointer = self.address(stmt.input, stmt.index)
                if count == 1:
                    self.put(stmt.names[0], self.load(pointer, stmt.input, 4))
                elif count in (2, 4):
                    vector = ir.VectorType(F32, count)
                    pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
                    value = self.load(pointer, stmt.input, 4 * count)
                    for i, name in enumerate(stmt.names):
                        self.put(name, self.builder.extract_element(value, integer(i)))
                else:
                    raise NotImplementedError(f'load width {count}')
            elif kind == 'Write':
                if stmt.swizzle != 'NONE':
                    raise NotImplementedError('swizzled writes are outside the open set')
                count = len(stmt.values)
                pointer = self.address(stmt.output, stmt.index)
                if stmt.atomic:
                    for i, name in enumerate(stmt.values):
                        pointer = self.address(stmt.output, stmt.index, i)
                        self.builder.atomic_rmw('fadd', pointer, self.get(name), 'monotonic')
                elif count == 1:
                    self.builder.store(self.get(stmt.values[0]), pointer, align=4)
                elif count in (2, 4):
                    vector = ir.VectorType(F32, count)
                    pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
                    value = ir.Constant(vector, ir.Undefined)
                    for i, name in enumerate(stmt.values):
                        value = self.builder.insert_element(value, self.get(name), integer(i))
                    self.builder.store(value, pointer, align=4 * count)
                else:
                    raise NotImplementedError(f'store width {count}')
            elif kind == 'Assign':
                previous = body[position - 1] if position else None
                source = reciprocal_sqrt_source(stmt, previous) if self.options.approximate_math else None
                if source is not None:
                    value = self.intrinsic('llvm.nvvm.rsqrt.approx.ftz.f', F32, [self.get(source)])
                else:
                    value = self.arithmetic(stmt.op.name, [self.get(a) for a in stmt.args])
                self.put(stmt.name, value)
            elif kind == 'Accum':
                base = self.get(stmt.base if stmt.base is not None else stmt.name)
                if self.reassociate and stmt.op.name == 'add':
                    value = self.builder.fadd(base, self.get(stmt.value), flags=('reassoc', 'nsz', 'contract'))
                else:
                    value = self.arithmetic(stmt.op.name, [base, self.get(stmt.value)])
                self.put(stmt.name, value)
            elif kind == 'Init':
                self.put(stmt.name, ir.Constant(F32, stmt.identity))
            elif kind == 'Let':
                value = self.expression(stmt.value)
                if value.type == I32:
                    value = self.builder.sitofp(value, F32)
                self.put(stmt.name, value)
            elif kind == 'Cond':
                self.divergent_depth += 1
                with self.builder.if_else(self.expression(stmt.cond)) as (then, otherwise):
                    with then:
                        self.body(stmt.body)
                    with otherwise:
                        self.body(stmt.else_body)
                self.divergent_depth -= 1
            elif kind == 'WarpShuffle':
                self.shuffle(stmt)
            elif kind == 'TreeHalve':
                self.tree(stmt)
            elif kind == 'Sync':
                self.synchronize(stmt.barrier_id, stmt.count, stmt.warp)
            elif kind == 'Smem':
                pass
            elif kind == 'CpAsyncCopy':
                self.async_copy(stmt)
            elif kind == 'CpAsyncCommit':
                self.assembly('cp.async.commit_group;', '~{memory}')
            elif kind == 'CpAsyncWait':
                self.assembly(f'cp.async.wait_group {stmt.group};', '~{memory}')
            else:
                raise NotImplementedError(f'Kernel IR statement {kind}')

    def generate(self):
        self.body(self.kernel.body)
        self.builder.ret_void()
        return str(self.module)

def compile_kernel(path: Path, *, arch='sm_89', opt_level=2, options=None) -> Compilation:
    if not re.fullmatch('sm_(?:8[069]|9[0]|1[012][0])', arch):
        raise ValueError(f'unsupported target {arch}; use sm_80, sm_86, sm_89, sm_90, sm_100, sm_110, or sm_120')
    if opt_level not in (0, 1, 2, 3):
        raise ValueError('optimization level must be 0, 1, 2, or 3')
    if options is None:
        options = Options()
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kernels = [n.op for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp']
    if len(kernels) != 1:
        raise ValueError('expected exactly one KernelOp in the input graph')
    kernel = kernels[0]
    metadata = json.loads(path.with_name('meta.json').read_text())
    launches = [k for k in metadata['kernels'] if k['entry'] == kernel.name]
    if len(launches) != 1:
        raise ValueError(f'meta.json must contain exactly one launch for {kernel.name}')
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    features = '+ptx87' if int(arch[3:]) >= 100 else '+ptx80'
    target = llvm.Target.from_triple('nvptx64-nvidia-cuda')
    machine = target.create_target_machine(cpu=arch, features=features, opt=opt_level)
    module = ir.Module(name=kernel.name)
    module.triple = 'nvptx64-nvidia-cuda'
    module.data_layout = str(machine.target_data)
    lowering = Lowering(kernel, launches[0], module, options)
    unoptimized = lowering.generate()
    with llvm.parse_assembly(unoptimized) as lowered:
        lowered.verify()
        tuning = llvm.PipelineTuningOptions(speed_level=opt_level)
        tuning.loop_vectorization = options.vector_width > 1
        tuning.slp_vectorization = opt_level >= 2
        with llvm.create_pass_builder(machine, tuning) as passes:
            with llvm.create_new_module_pass_manager() as promotion:
                promotion.add_sroa_pass()
                promotion.run(lowered, passes)
            with passes.getModulePassManager() as optimizer:
                optimizer.run(lowered, passes)
            if opt_level:
                with llvm.create_new_module_pass_manager() as cleanup:
                    if lowering.uses_cooperative_dots:
                        cleanup.add_loop_strength_reduce_pass()
                    cleanup.add_new_gvn_pass()
                    cleanup.add_instruction_combine_pass()
                    cleanup.add_simplify_cfg_pass()
                    cleanup.run(lowered, passes)
        lowered.verify()
        optimized = str(lowered)
        ptx = machine.emit_assembly(lowered)
    return Compilation(ptx, unoptimized, optimized)
