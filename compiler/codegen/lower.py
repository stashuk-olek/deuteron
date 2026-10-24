import contextlib
import math
from typing import Any
from llvmlite import ir
from compiler.analysis import STATIC_SHARED_LIMIT, walk
from compiler.codegen.tensor import TensorEmitter
from compiler.options import Options
from compiler.passes import run_passes
from compiler.passes.regstage import stage_register_loads
from compiler.passes.hybrid_normalizer import hybrid_normalizer
from compiler.passes.tailtree import linearize_small_trees
from compiler.passes.regforward import _no_writes_to
from compiler.passes.recurrence import RecurrenceEmitter
from compiler.storage import half_storage_types
from compiler.validation import validate_kernel_launch
from compiler.transforms import default_unroll, plain_reduction, reciprocal_sqrt_source, sharing_width, striped_reduction
I1: Any = ir.IntType(1)
I8: Any = ir.IntType(8)
I32: Any = ir.IntType(32)
I64: Any = ir.IntType(64)
F32: Any = ir.FloatType()
F16: Any = ir.HalfType()
VOID: Any = ir.VoidType()

def integer(value: Any) -> Any:
    return ir.Constant(I32, int(value.value if hasattr(value, 'as_static') else value))

class Lowering:

    def __init__(self, kernel: Any, launch: Any, module: Any, options: Options) -> None:
        specs = validate_kernel_launch(kernel, launch)
        self.storage_types = half_storage_types(kernel, launch)
        self.hybrid_max: Any = None
        hybrid = hybrid_normalizer(kernel, launch) if options.approximate_math else None
        if hybrid is not None:
            kernel, self.hybrid_max = hybrid
        self.packed_operands: tuple = ()
        staged = stage_register_loads(kernel, launch) if options.contraction_precision == 'bf16x3' else None
        if staged is not None:
            kernel, self.packed_operands = staged
        self.coalesced: set[int] = {id(loop) for tile in kernel.body if type(tile).__name__ == 'Tile' for loop in tile.body if options.coalesced_reductions and striped_reduction(loop, tile, launch['block'][0])}
        if options.tail_linearize:
            kernel = linearize_small_trees(kernel, launch, options) or kernel
        self.kernel = kernel
        self.launch = launch
        self.module = module
        self.options = options
        self.tensors: list[Any] = []
        self.recurrences: dict[Any, Any] = {}
        self.tile_stmt: Any = None
        self.uniform_axes: set[Any] = set()
        self.divergent_depth: int = 0
        self.reassociate: bool = False
        self.striped: bool = False
        self.slots: dict[str, Any] = {}
        self.buffers: dict[str, Any] = {}
        self.shapes: dict[str, Any] = {}
        self.readonly = {a['name'] for a in launch['args'] if not a.get('output')}
        signature = ir.FunctionType(VOID, [(F16 if specs[name]['dtype'] == 'f16' else F32).as_pointer(1) for name in launch['arg_order']])
        self.function: Any = ir.Function(module, signature, kernel.name)
        self.function.calling_convention = 'ptx_kernel'
        module.add_named_metadata('nvvm.annotations', [self.function, 'kernel', integer(1)])
        module.add_named_metadata('nvvm.annotations', [self.function, 'maxntidx', integer(launch['block'][0])])
        self.builder: Any = ir.IRBuilder(self.function.append_basic_block('entry'))
        for arg, name in zip(self.function.args, launch['arg_order'], strict=True):
            arg.name = name
            arg.attributes.add('noalias')
            arg.attributes.align = 16
            arg.attributes.dereferenceable = math.prod(specs[name]['shape']) * (2 if specs[name]['dtype'] == 'f16' else 4)
            self.buffers[name] = arg
            self.shapes[name] = tuple(specs[name]['shape'])
        for stmt in walk(kernel.body):
            kind = type(stmt).__name__
            if kind in ('Load', 'Assign', 'Accum', 'Init', 'Let', 'Select'):
                dtype = getattr(stmt, 'dtype', None)
                if dtype is not None and dtype.name != 'f32' and not (kind == 'Load' and dtype.name == 'f16' and self.storage_types is not None):
                    raise NotImplementedError(f'{kind}: unsupported dtype {dtype.name}')
                names = stmt.names if kind == 'Load' else (stmt.name,)
                for name in names:
                    self.allocate(name, F16 if self.storage_types is not None and self.storage_types.get(name) == 'f16' else F32)
            if kind in ('WarpShuffle', 'TreeHalve'):
                for name in (*stmt.state, *stmt.state_b):
                    self.allocate(name, F32)
            if kind in ('Loop', 'StridedLoop'):
                self.allocate(stmt.axis.name, I32)
                if self.options.coalesced_reductions and plain_reduction(stmt):
                    accum = next((s for s in stmt.body if type(s).__name__ == 'Accum'))
                    self.allocate('__vector4_' + accum.name, ir.VectorType(F32, 4))
                    self.allocate('__vector8_' + accum.name, ir.VectorType(F32, 8))
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
        hints = run_passes(kernel, launch, options)
        for plan in hints.recurrences:
            try:
                self.recurrences[id(plan.tile)] = RecurrenceEmitter(self, plan)
            except ValueError:
                continue
        self.tensors = [TensorEmitter(self, plan, options.contraction_precision) for plan in hints.contractions]

    def allocate(self, name: Any, dtype: Any) -> None:
        if name not in self.slots:
            self.slots[name] = self.builder.alloca(dtype, name=name + '.slot')
        elif self.slots[name].type.pointee != dtype:
            raise ValueError(f'inconsistent type for {name}')

    def get(self, name: Any) -> Any:
        if name not in self.slots:
            raise ValueError(f'unbound value {name}')
        return self.builder.load(self.slots[name], name=name)

    def put(self, name: Any, value: Any) -> None:
        self.builder.store(value, self.slots[name])

    def intrinsic(self, name: Any, result: Any, args: Any) -> Any:
        signature = ir.FunctionType(result, [a.type for a in args])
        function = self.module.globals.get(name)
        if function is None:
            function = ir.Function(self.module, signature, name)
        call = self.builder.call(function, args)
        if self.options.warp_willreturn and name in ('llvm.nvvm.activemask', 'llvm.nvvm.shfl.sync.bfly.i32'):
            set.add(call.attributes, 'willreturn')
        return call

    def assembly(self, text: Any, constraints: Any, result: Any = VOID, args: Any = ()) -> Any:
        signature = ir.FunctionType(result, [a.type for a in args])
        asm = ir.InlineAsm(signature, text, constraints, side_effect=True)
        return self.builder.call(asm, args)

    def shared_memory(self) -> None:
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
        pool: Any = None
        if dynamic:
            if size > self.launch['smem_bytes']:
                raise ValueError(f"launch provides {self.launch['smem_bytes']} shared bytes; IR needs {size}")
            pool: Any = ir.GlobalVariable(self.module, ir.ArrayType(I8, 0), 'dynamic_shared', addrspace=3)
            pool.linkage = 'external'
            pool.align = max((s.align or 4 for s in declarations.values()))
        for name, stmt in declarations.items():
            self.shapes[name] = tuple(stmt.extents)
            if dynamic:
                pointer = self.builder.gep(pool, [integer(0), integer(offsets[name])])
                self.buffers[name] = self.builder.bitcast(pointer, F32.as_pointer(3))
            else:
                array: Any = ir.GlobalVariable(self.module, ir.ArrayType(F32, math.prod(stmt.extents)), name, addrspace=3)
                array.linkage = 'internal'
                array.initializer = ir.Constant(array.value_type, ir.Undefined)
                array.align = max(4, stmt.align)
                self.buffers[name] = self.builder.gep(array, [integer(0), integer(0)])

    def expression(self, expr: Any) -> Any:
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
        if kind == 'CastExpr':
            value = self.expression(expr.expr)
            if expr.dtype == 'int':
                return self.builder.fptosi(value, I32) if value.type == F32 else value
            return self.builder.sitofp(value, F32) if value.type == I32 else value
        if kind == 'FlatIndex':
            return self.offset(expr.buffer, expr.index)
        if kind == 'Builtin':
            return self.builtin(expr.name)
        if kind == 'FuncCallExpr':
            values = [self.expression(a) for a in expr.args]
            if all((v.type == F32 for v in values)):
                return self.arithmetic(expr.name, values)
            if all((v.type == I32 for v in values)) and len(values) == 2:
                method = {'add': 'add', 'subtract': 'sub', 'multiply': 'mul'}.get(expr.name)
                if method:
                    return getattr(self.builder, method)(*values)
        raise NotImplementedError(f'expression {expr!r}')

    def offset(self, buffer: Any, index: Any) -> Any:
        shape = self.shapes[buffer]
        if not index and math.prod(shape) == 1:
            shape = ()
        if len(index) != len(shape):
            raise ValueError(f'{buffer}: index rank {len(index)} does not match shape {shape}')
        offset = integer(0)
        for coordinate, extent in zip(index, shape, strict=True):
            value = coordinate if isinstance(coordinate, ir.Value) else self.expression(coordinate)
            offset = self.builder.add(self.builder.mul(offset, integer(extent)), value)
        return offset

    def address(self, buffer: Any, index: Any, extra: Any = 0) -> Any:
        for emitter in self.tensors:
            index = emitter.remap(buffer, index)
        offset = self.offset(buffer, index)
        if extra:
            offset = self.builder.add(offset, integer(extra))
        return self.builder.gep(self.buffers[buffer], [self.builder.sext(offset, I64)])

    def builtin(self, name: Any) -> Any:
        if name == 'threadIdx.x':
            return self.tid
        if name == 'blockIdx.x':
            return self.intrinsic('llvm.nvvm.read.ptx.sreg.ctaid.x', I32, [])
        if name == 'blockDim.x':
            return self.intrinsic('llvm.nvvm.read.ptx.sreg.ntid.x', I32, [])
        if name in ('threadIdx.y', 'threadIdx.z', 'blockIdx.y', 'blockIdx.z'):
            return integer(0)
        if name in ('blockDim.y', 'blockDim.z'):
            return integer(1)
        raise NotImplementedError(f'gpu builtin {name}')

    def load(self, pointer: Any, buffer: Any, alignment: Any) -> Any:
        value = self.builder.load(pointer, align=alignment)
        if buffer in self.readonly:
            value.set_metadata('invariant.load', self.module.add_metadata([]))
        return value

    def convert(self, value: Any, dtype: Any) -> Any:
        if value.type == dtype:
            return value
        if value.type == F16 and dtype == F32:
            return self.builder.fpext(value, F32)
        if value.type == F32 and dtype == F16:
            return self.builder.fptrunc(value, F16)
        raise NotImplementedError(f'unsupported value conversion {value.type} to {dtype}')

    def arithmetic(self, op: Any, args: Any) -> Any:
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
        if op == 'minimum':
            return self.intrinsic('llvm.minnum.f32', F32, args)
        if op == 'rsqrt':
            if self.options.approximate_math:
                return self.intrinsic('llvm.nvvm.rsqrt.approx.ftz.f', F32, args)
            return self.builder.fdiv(ir.Constant(F32, 1.0), self.intrinsic('llvm.sqrt.f32', F32, args))
        if op == 'exp_fast':
            return self.arithmetic('exp', args)
        if op == 'negative':
            return self.builder.fneg(args[0])
        if op == 'absolute':
            return self.intrinsic('llvm.fabs.f32', F32, args)
        raise NotImplementedError(f'elementwise operation {op}')

    def register_reduction(self, stmt: Any, start: Any, step: Any, width: int, trips: int) -> None:
        load = stmt.body[0]
        self.put(stmt.axis.name, start)
        source = self.address(load.input, load.index)
        stride = self.launch['block'][0] * width * 4
        chunk = 32 // width
        code = ['{', '.reg .b64 g;', '.reg .b32 left;', '.reg .pred p;', '.reg .f32 a<32>,x<32>,y<32>;', 'mov.u64 g, $1;']
        def copy(prefix: str) -> None:
            for j in range(chunk):
                for k in range(width // 4):
                    regs = ','.join(f'{prefix}{j * width + n}' for n in range(k * 4, k * 4 + 4))
                    code.append(f'ld.global.nc.v4.f32 {{{regs}}}, [g+{j * stride + k * 16}];')
            code.append(f'add.u64 g,g,{chunk * stride};')
        def consume(prefix: str) -> None:
            for j in range(32):
                code.append(f'fma.rn.f32 a{j},{prefix}{j},{prefix}{j},a{j};' if len(stmt.body) == 3 else f'add.f32 a{j},a{j},{prefix}{j};')
        copy('a')
        if len(stmt.body) == 3:
            code += [f'mul.f32 a{j},a{j},a{j};' for j in range(32)]
        copy('x')
        code += [f'mov.u32 left,{trips // chunk // 2 - 1};', 'registerloop:', r'.pragma \22nounroll\22;']
        copy('y')
        consume('x')
        copy('x')
        consume('y')
        code += ['sub.u32 left,left,1;', 'setp.ne.u32 p,left,0;', '@p bra registerloop;']
        consume('x')
        distance = 16
        while distance:
            code += [f'add.f32 a{j},a{j},a{j+distance};' for j in range(distance)]
            distance //= 2
        code += ['mov.f32 $0,a0;', '}']
        result = self.assembly('\n'.join(code), '=f,l,~{memory}', F32, (source,))
        self.put(stmt.body[-1].name, result)
        self.put(stmt.axis.name, self.builder.add(start, self.builder.mul(step, integer(trips))))

    def loop(self, stmt: Any) -> None:
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
        if self.options.reduction_sharing and self.tile_stmt is not None and (self.divergent_depth == 0) and (self.launch['block'][0] % 32 == 0):
            cooperative = sharing_width(stmt, self.tile_stmt, self.uniform_axes)
        if cooperative > 1:
            self.striped = True
            start = self.builder.urem(self.get('lane'), integer(cooperative))
            step = integer(cooperative)
        coalesced = self.tile_stmt is not None and id(stmt) in self.coalesced
        width = 4
        trips = 0
        count: Any = None
        if coalesced:
            count = self.options.unroll or default_unroll(stmt, self.tile_stmt, self.uniform_axes)
            iters = stmt.axis.extent.value // (self.launch['block'][0] * 4)
            if iters > count >= iters // 2 and stmt.axis.extent.value % (self.launch['block'][0] * 8) == 0:
                width = 8
            trips = stmt.axis.extent.value // (self.launch['block'][0] * width)
            if trips <= 128:
                count = max(count, trips)
        if coalesced:
            start = self.builder.mul(start, integer(width))
            step = self.builder.mul(step, integer(width))
        if coalesced and not self.options.unroll and trips >= 32 and trips % (64 // width) == 0 and stmt.body[-1].op.name == 'add' and (len(stmt.body) == 2 or stmt.body[1].op.name == 'multiply') and stmt.body[0].input in self.readonly and self.buffers[stmt.body[0].input].type.addrspace == 1 and _no_writes_to(self.tile_stmt, stmt.body[0].input):
            self.register_reduction(stmt, start, step, width, trips)
            return
        vectorize = self.options.vector_width > 1 and plain_reduction(stmt) and (not coalesced)
        vector_name: Any = None
        if coalesced:
            vector_name = '__vector' + str(width) + '_' + stmt.body[-1].name
            self.put(vector_name, ir.Constant(ir.VectorType(F32, width), [0.0] * width))
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
            vector = ir.VectorType(F32, width)
            pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
            values = self.load(pointer, load.input, 4 * width)
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
        if not coalesced:
            count = self.options.unroll or default_unroll(stmt, self.tile_stmt, self.uniform_axes)
        if not self.options.unroll and any(t.packed_shared and t.plan.outer is stmt for t in self.tensors):
            count = 2
        hint = ['llvm.loop.unroll.count', integer(count)] if count > 1 else ['llvm.loop.unroll.disable']
        metadata: Any = ir.MDValue(self.module, (), name=str(len(self.module.metadata)))
        hints = [self.module.add_metadata(hint)]
        if vectorize:
            hints += [self.module.add_metadata(['llvm.loop.vectorize.enable', ir.Constant(I1, 1)]), self.module.add_metadata(['llvm.loop.vectorize.width', integer(self.options.vector_width)]), self.module.add_metadata(['llvm.loop.interleave.count', integer(1)])]
        metadata.operands = (metadata, *hints)
        latch.set_metadata('llvm.loop', metadata)
        self.builder.position_at_end(exit_block)
        if coalesced:
            partial = self.intrinsic(f'llvm.vector.reduce.fadd.v{width}f32', F32, [ir.Constant(F32, 0.0), self.get(vector_name)])
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

    def tile(self, stmt: Any) -> None:
        if stmt.aux_threads:
            raise NotImplementedError('auxiliary threads are outside the open set')
        if stmt.block_threads and stmt.block_threads != self.launch['block'][0]:
            raise ValueError('Tile block_threads disagrees with the launch')
        block = self.intrinsic('llvm.nvvm.read.ptx.sreg.ctaid.x', I32, [])
        size = math.prod((a.extent.value for a in stmt.axes))
        exact = size == self.launch['grid'][0] * self.launch['block'][0]
        split, width = len(stmt.axes) - 1, stmt.axes[-1].extent.value
        while split and width < self.launch['block'][0] and (not exact or 0 < size <= 2**32 and all(a.extent.value > 0 for a in stmt.axes)):
            split -= 1
            width *= stmt.axes[split].extent.value
        direct = not (stmt.raster_group and stmt.raster_axes) and width == self.launch['block'][0] and (exact or (0 < size < 2**32 and self.launch['grid'][0] * self.launch['block'][0] <= 2**32 and all(a.extent.value > 0 for a in stmt.axes)))
        if direct:
            outer = block
            gid = None
        else:
            threads = self.intrinsic('llvm.nvvm.read.ptx.sreg.ntid.x', I32, [])
            gid = self.builder.add(self.builder.mul(block, threads), self.tid)
            outer = gid
        guard = contextlib.nullcontext() if exact else self.builder.if_then(self.builder.icmp_unsigned('<', gid if gid is not None else block, integer(size if gid is not None else size // self.launch['block'][0])))
        with guard:
            leading = stmt.axes[:split] if gid is None else stmt.axes
            if gid is None and len(leading) == 1 and leading[0].extent.value == self.launch['grid'][0]:
                self.put(leading[0].name, outer)
            else:
                stride = size if gid is not None else size // self.launch['block'][0]
                for axis in leading:
                    extent = axis.extent.value
                    stride //= extent
                    value = self.builder.urem(self.builder.udiv(outer, integer(stride)), integer(extent))
                    self.put(axis.name, value)
            if gid is None:
                stride = self.launch['block'][0]
                for axis in stmt.axes[split:]:
                    stride //= axis.extent.value
                    value = self.tid if split == len(stmt.axes) - 1 else self.builder.urem(self.builder.udiv(self.tid, integer(stride)), integer(axis.extent.value))
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
            emitter = self.recurrences.get(id(stmt))
            if emitter is not None:
                emitter.emit()
            else:
                self.body(stmt.body)
            self.tile_stmt = old_tile

    def synchronize(self, barrier_id: Any = 0, count: Any = None, warp: Any = False) -> None:
        if warp:
            mask = self.intrinsic('llvm.nvvm.activemask', I32, [])
            self.intrinsic('llvm.nvvm.bar.warp.sync', VOID, [mask])
        elif barrier_id:
            if count is None:
                raise ValueError('named barrier requires a thread count')
            self.assembly(f'bar.sync {barrier_id}, {count};', '~{memory}')
        else:
            self.intrinsic('llvm.nvvm.barrier0', VOID, [])

    def ordered_maximum(self, value: Any) -> Any:
        bits = self.builder.bitcast(value, I32)
        mask = self.builder.lshr(self.builder.ashr(bits, integer(31)), integer(1))
        ordered = self.builder.xor(bits, mask)
        result = self.assembly('redux.sync.max.s32 $0, $1, $2;', '=r,r,r', I32, (ordered, integer(-1)))
        result.attributes.add('convergent')
        mask = self.builder.lshr(self.builder.ashr(result, integer(31)), integer(1))
        return self.builder.bitcast(self.builder.xor(result, mask), F32)

    def shuffle(self, stmt: Any) -> None:
        if stmt.length < 1 or stmt.length > 32 or stmt.length & stmt.length - 1:
            raise ValueError('shuffle length must be a power of two up to 32')
        if stmt is self.hybrid_max:
            self.put(stmt.state[0], self.ordered_maximum(self.get(stmt.state[0])))
            return
        mask = self.intrinsic('llvm.nvvm.activemask', I32, [])
        distance = stmt.length // 2
        while distance:
            for state, other in zip(stmt.state, stmt.state_b, strict=True):
                bits = self.builder.bitcast(self.get(state), I32)
                value = self.intrinsic('llvm.nvvm.shfl.sync.bfly.i32', I32, [mask, bits, integer(distance), integer(31)])
                self.put(other, self.builder.bitcast(value, F32))
            self.body(stmt.combine_states)
            distance //= 2

    def tree(self, stmt: Any) -> None:
        if stmt.length < 1 or stmt.length & stmt.length - 1:
            raise ValueError('tree length must be a power of two')
        if self.options.tree_specialization and stmt.tid_var == 'warp' and (stmt.inner is None) and (stmt.length <= 32):
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

    def async_copy(self, stmt: Any) -> None:
        if stmt.swizzle != 'NONE':
            raise NotImplementedError('shared-memory swizzling is outside the open set')
        if stmt.nbytes not in (4, 8, 16):
            raise ValueError('cp.async copy size must be 4, 8, or 16')
        index = [self.expression(x) for x in stmt.smem_index]
        if stmt.lane_index is not None:
            index = [self.builder.add(x, self.expression(y)) for x, y in zip(index, stmt.lane_index, strict=True)]
        destination = self.builder.ptrtoint(self.address(stmt.smem, index), I32)
        source = self.builder.ptrtoint(self.address(stmt.src, stmt.src_index), I64)
        policy = 'cg' if stmt.nbytes == 16 else 'ca'
        self.assembly(f'cp.async.{policy}.shared.global [$0], [$1], {stmt.nbytes};', 'r,l,~{memory}', args=(destination, source))

    def body(self, body: Any) -> None:
        for position, stmt in enumerate(body):
            kind = type(stmt).__name__
            replaced = False
            for emitter in self.tensors:
                if stmt is emitter.plan.loop:
                    emitter.emit()
                    replaced = True
                elif kind == 'Write' and emitter.emit_store(stmt):
                    replaced = True
            if replaced:
                continue
            if kind == 'Tile':
                self.tile(stmt)
            elif kind in ('Loop', 'StridedLoop'):
                self.loop(stmt)
            elif kind == 'Load':
                count = len(stmt.names)
                pointer = self.address(stmt.input, stmt.index)
                dtype = pointer.type.pointee
                alignment = (2 if dtype == F16 else 4) if self.storage_types is not None else 4 * count
                if count == 1:
                    self.put(stmt.names[0], self.load(pointer, stmt.input, alignment))
                elif count in (2, 4):
                    vector = ir.VectorType(dtype, count)
                    pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
                    value = self.load(pointer, stmt.input, alignment)
                    for i, name in enumerate(stmt.names):
                        self.put(name, self.builder.extract_element(value, integer(i)))
                else:
                    raise NotImplementedError(f'load width {count}')
            elif kind == 'Write':
                if stmt.swizzle != 'NONE':
                    raise NotImplementedError('swizzled writes are outside the open set')
                if not stmt.atomic and any(emitter.packed_store(stmt.output, [self.expression(v) for v in stmt.index], [self.get(v) for v in stmt.values]) for emitter in self.tensors):
                    continue
                count = len(stmt.values)
                pointer = self.address(stmt.output, stmt.index)
                dtype = pointer.type.pointee
                alignment = (2 if dtype == F16 else 4) if self.storage_types is not None else 4 * count
                if stmt.atomic:
                    for i, name in enumerate(stmt.values):
                        pointer = self.address(stmt.output, stmt.index, i)
                        self.builder.atomic_rmw('fadd', pointer, self.get(name), 'monotonic')
                elif count == 1:
                    self.builder.store(self.convert(self.get(stmt.values[0]), dtype), pointer, align=alignment)
                elif count in (2, 4):
                    vector = ir.VectorType(dtype, count)
                    pointer = self.builder.bitcast(pointer, vector.as_pointer(pointer.type.addrspace))
                    value = ir.Constant(vector, ir.Undefined)
                    for i, name in enumerate(stmt.values):
                        value = self.builder.insert_element(value, self.convert(self.get(name), dtype), integer(i))
                    self.builder.store(value, pointer, align=alignment)
                else:
                    raise NotImplementedError(f'store width {count}')
            elif kind == 'Assign':
                previous = body[position - 1] if position else None
                source = reciprocal_sqrt_source(stmt, previous) if self.options.approximate_math else None
                if source is not None:
                    value = self.intrinsic('llvm.nvvm.rsqrt.approx.ftz.f', F32, [self.convert(self.get(source), F32)])
                else:
                    value = self.arithmetic(stmt.op.name, [self.convert(self.get(a), F32) for a in stmt.args])
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
            elif kind == 'Select':
                value = self.get(stmt.branches[-1].value)
                for branch in reversed(stmt.branches[:-1]):
                    value = self.builder.select(self.expression(branch.select), self.get(branch.value), value)
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

    def generate(self) -> str:
        self.body(self.kernel.body)
        self.builder.ret_void()
        return str(self.module)

