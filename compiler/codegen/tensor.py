from llvmlite import ir
from compiler.analysis import known_multiple
from compiler.transforms import shared_layout
F32 = ir.FloatType()
I32 = ir.IntType(32)

def integer(value):
    return ir.Constant(I32, int(value))

class TensorEmitter:

    def __init__(self, lowering, plan, mode='bf16x3'):
        if mode not in ('tf32x3', 'bf16x3'):
            raise ValueError(f'unsupported contraction precision {mode}')
        self.lowering = lowering
        self.plan = plan
        self.mode = mode
        self.warps_n = plan.cols // 8
        self.active_threads = plan.rows // 16 * self.warps_n * 32
        self.accumulators = [lowering.builder.alloca(F32, name='tensor.acc') for _ in range(4)]
        for slot in self.accumulators:
            lowering.builder.store(ir.Constant(F32, 0.0), slot)

    def coordinates(self):
        lower = self.lowering
        b = lower.builder
        lane, warp = (lower.get('lane'), lower.get('warp'))
        row = b.add(b.mul(b.udiv(warp, integer(self.warps_n)), integer(16)), b.udiv(lane, integer(4)))
        col = b.add(b.mul(b.urem(warp, integer(self.warps_n)), integer(8)), b.mul(b.urem(lane, integer(4)), integer(2)))
        return (row, col)

    def remap(self, buffer, index):
        plan = self.plan
        if buffer not in (plan.lhs_buffer, plan.rhs_buffer):
            return index
        lower = self.lowering
        b = lower.builder
        row, col = [v if isinstance(v, ir.Value) else lower.expression(v) for v in index]
        columns = lower.shapes[buffer][1]
        role = 'row' if buffer == plan.lhs_buffer else 'col'
        shift, mask, halve = shared_layout(columns, role, self.mode)
        mixing = b.lshr(row, integer(1)) if halve else row
        if mask:
            col = b.xor(col, b.shl(b.and_(mixing, integer(mask)), integer(shift)))
        return [row, col]

    def tf32(self, value):
        signature = ir.FunctionType(I32, [F32])
        asm = ir.InlineAsm(signature, 'cvt.rna.tf32.f32 $0, $1;', '=r,f', side_effect=False)
        return self.lowering.builder.call(asm, [value])

    def bf16(self, values):
        b = self.lowering.builder
        signature = ir.FunctionType(I32, [F32, F32])
        asm = ir.InlineAsm(signature, 'cvt.rn.bf16x2.f32 $0, $1, $2;', '=r,f,f', side_effect=False)
        packed, restored = ([], [])
        for i in range(0, len(values), 2):
            value = b.call(asm, [values[i + 1], values[i]])
            packed.append(value)
            restored.extend((b.bitcast(b.shl(value, integer(16)), F32), b.bitcast(b.and_(value, integer(-65536)), F32)))
        return (packed, restored)

    def mma(self, a_values, b_values, accumulators):
        b = self.lowering.builder
        result_type = ir.LiteralStructType([F32] * 4)
        signature = ir.FunctionType(result_type, [I32] * 6 + [F32] * 4)
        instruction = 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32' if self.mode == 'bf16x3' else 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'
        text = instruction + ' {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};'
        asm = ir.InlineAsm(signature, text, '=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3', side_effect=False)
        value = b.call(asm, a_values + b_values + accumulators)
        value.attributes.add('convergent')
        return [b.extract_value(value, i) for i in range(4)]

    def active(self, emit):
        lower = self.lowering
        if self.active_threads < lower.launch['block'][0]:
            with lower.builder.if_then(lower.builder.icmp_unsigned('<', lower.tid, integer(self.active_threads))):
                emit()
        else:
            emit()

    def emit(self):
        self.active(self._emit)

    def _emit(self):
        lower, plan = (self.lowering, self.plan)
        b = lower.builder
        row, _ = self.coordinates()
        lane, warp = (lower.get('lane'), lower.get('warp'))
        packed = self.mode == 'bf16x3'
        quad_k = b.mul(b.urem(lane, integer(4)), integer(2 if packed else 1))
        column = b.add(b.mul(b.urem(warp, integer(self.warps_n)), integer(8)), b.udiv(lane, integer(4)))
        lhs_base = [lower.expression(v) for v in plan.lhs_base]
        rhs_base = [lower.expression(v) for v in plan.rhs_base]
        accumulators = [b.load(slot) for slot in self.accumulators]

        def load_lhs(k_offset, row_offset):
            if k_offset >= plan.depth:
                return ir.Constant(F32, 0.0)
            indices = [b.add(lhs_base[0], b.add(row, integer(row_offset))), b.add(lhs_base[1], b.add(quad_k, integer(k_offset)))]
            return lower.load(lower.address(plan.lhs_buffer, indices), plan.lhs_buffer, 4)

        def load_lhs_pair(k_offset, row_offset):
            if k_offset >= plan.depth:
                return [ir.Constant(F32, 0.0)] * 2
            if known_multiple(plan.lhs_base[1]) % 2:
                return [load_lhs(k_offset + i, row_offset) for i in range(2)]
            indices = [b.add(lhs_base[0], b.add(row, integer(row_offset))), b.add(lhs_base[1], b.add(quad_k, integer(k_offset)))]
            pointer = lower.address(plan.lhs_buffer, indices)
            vector_type = ir.VectorType(F32, 2)
            vector = b.load(b.bitcast(pointer, vector_type.as_pointer(pointer.type.addrspace)), align=4)
            return [b.extract_element(vector, integer(i)) for i in range(2)]

        def load_rhs(k_offset):
            if k_offset >= plan.depth:
                return ir.Constant(F32, 0.0)
            indices = [b.add(rhs_base[0], b.add(quad_k, integer(k_offset))), b.add(rhs_base[1], column)]
            return lower.load(lower.address(plan.rhs_buffer, indices), plan.rhs_buffer, 4)
        for k in range(0, plan.depth, 16 if packed else 8):
            if packed:
                lhs_values = [value for q in (0, 8) for m in (0, 8) for value in load_lhs_pair(k + q, m)]
                rhs_values = [load_rhs(k + q + v) for q in (0, 8) for v in (0, 1)]
                lhs_high, lhs_restored = self.bf16(lhs_values)
                rhs_high, rhs_restored = self.bf16(rhs_values)
                lhs_low, _ = self.bf16([b.fsub(v, h) for v, h in zip(lhs_values, lhs_restored, strict=True)])
                rhs_low, _ = self.bf16([b.fsub(v, h) for v, h in zip(rhs_values, rhs_restored, strict=True)])
            else:
                lhs_values = [load_lhs(k + q, m) for q in (0, 4) for m in (0, 8)]
                rhs_values = [load_rhs(k + q) for q in (0, 4)]
                lhs_high = [self.tf32(v) for v in lhs_values]
                rhs_high = [self.tf32(v) for v in rhs_values]
                lhs_low = [self.tf32(b.fsub(v, b.bitcast(h, F32))) for v, h in zip(lhs_values, lhs_high, strict=True)]
                rhs_low = [self.tf32(b.fsub(v, b.bitcast(h, F32))) for v, h in zip(rhs_values, rhs_high, strict=True)]
            accumulators = self.mma(lhs_low, rhs_high, accumulators)
            accumulators = self.mma(lhs_high, rhs_low, accumulators)
            accumulators = self.mma(lhs_high, rhs_high, accumulators)
        for slot, value in zip(self.accumulators, accumulators, strict=True):
            b.store(value, slot)

    def emit_store(self, statement):
        if not any((statement is candidate for candidate in self.plan.writes)):
            return False
        if statement is not self.plan.writes[0]:
            return True
        self.active(self._store)
        return True

    def _store(self):
        lower, plan = (self.lowering, self.plan)
        b = lower.builder
        row, col = self.coordinates()
        row = b.add(row, lower.expression(plan.output_base[0]))
        col = b.add(col, lower.expression(plan.output_base[1]))
        for i, slot in enumerate(self.accumulators):
            pointer = lower.address(plan.output_buffer, [b.add(row, integer(i // 2 * 8)), b.add(col, integer(i % 2))])
            value = b.load(slot)
            if plan.atomic:
                b.atomic_rmw('fadd', pointer, value, 'monotonic')
            else:
                b.store(value, pointer, align=4)
