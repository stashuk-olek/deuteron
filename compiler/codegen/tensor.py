from typing import Any
from llvmlite import ir
from compiler.analysis import known_multiple
from compiler.options import Contraction
from compiler.transforms import shared_layout
F32: Any = ir.FloatType()
I32: Any = ir.IntType(32)

def integer(value: Any) -> Any:
    return ir.Constant(I32, int(value))

class TensorEmitter:

    def __init__(self, lowering: Any, plan: Contraction, mode: str = 'bf16x3') -> None:
        if mode not in ('tf32x3', 'bf16x3'):
            raise ValueError(f'unsupported contraction precision {mode}')
        self.lowering = lowering
        self.plan = plan
        self.mode = mode
        self.packed_shared = mode == 'bf16x3' and (plan.lhs_buffer, plan.rhs_buffer) == lowering.packed_operands
        self.warps_n = plan.cols // 8
        self.active_threads = plan.rows // 16 * self.warps_n * 32
        self.accumulators = [lowering.builder.alloca(F32, name='tensor.acc') for _ in range(4)]
        for slot in self.accumulators:
            lowering.builder.store(ir.Constant(F32, 0.0), slot)

    def coordinates(self) -> Any:
        lower = self.lowering
        b = lower.builder
        lane, warp = (lower.get('lane'), lower.get('warp'))
        row = b.add(b.mul(b.udiv(warp, integer(self.warps_n)), integer(16)), b.udiv(lane, integer(4)))
        col = b.add(b.mul(b.urem(warp, integer(self.warps_n)), integer(8)), b.mul(b.urem(lane, integer(4)), integer(2)))
        return (row, col)

    def remap(self, buffer: Any, index: Any) -> Any:
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

    def tf32(self, value: Any) -> Any:
        signature = ir.FunctionType(I32, [F32])
        asm = ir.InlineAsm(signature, 'cvt.rna.tf32.f32 $0, $1;', '=r,f', side_effect=False)
        return self.lowering.builder.call(asm, [value])

    def bf16(self, values: Any) -> Any:
        b = self.lowering.builder
        signature = ir.FunctionType(I32, [F32, F32])
        asm = ir.InlineAsm(signature, 'cvt.rn.bf16x2.f32 $0, $1, $2;', '=r,f,f', side_effect=False)
        packed, restored = ([], [])
        for i in range(0, len(values), 2):
            value = b.call(asm, [values[i + 1], values[i]])
            packed.append(value)
            restored.extend((b.bitcast(b.shl(value, integer(16)), F32), b.bitcast(b.and_(value, integer(-65536)), F32)))
        return (packed, restored)

    def mma(self, a_values: Any, b_values: Any, accumulators: Any) -> Any:
        b = self.lowering.builder
        result_type = ir.LiteralStructType([F32] * 4)
        signature = ir.FunctionType(result_type, [I32] * 6 + [F32] * 4)
        instruction = 'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32' if self.mode == 'bf16x3' else 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'
        text = instruction + ' {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};'
        asm = ir.InlineAsm(signature, text, '=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3', side_effect=False)
        value = b.call(asm, a_values + b_values + accumulators)
        value.attributes.add('convergent')
        return [b.extract_value(value, i) for i in range(4)]

    def active(self, emit: Any) -> None:
        lower = self.lowering
        if self.active_threads < lower.launch['block'][0]:
            with lower.builder.if_then(lower.builder.icmp_unsigned('<', lower.tid, integer(self.active_threads))):
                emit()
        else:
            emit()

    def emit(self) -> None:
        self.active(self._emit_packed if self.packed_shared else self._emit)

    def packed_address(self, buffer: Any, index: Any, low: Any = False) -> Any:
        lower, plan = self.lowering, self.plan
        b = lower.builder
        row, col = index
        rows = plan.rows if buffer == plan.lhs_buffer else plan.depth
        columns = lower.shapes[buffer][1]
        stage = b.udiv(row, integer(rows))
        row = b.urem(row, integer(rows))
        mask = min(7, (columns - 1) // 8)
        col = b.xor(col, b.shl(b.and_(row, integer(mask)), integer(3)))
        offset = b.add(b.mul(stage, integer(rows * columns * 2)), b.add(b.mul(row, integer(columns)), col))
        if low:
            offset = b.add(offset, integer(rows * columns))
        pointer = b.bitcast(lower.buffers[buffer], ir.IntType(16).as_pointer(3))
        return b.gep(pointer, [offset])

    def packed_store(self, buffer: Any, index: Any, values: Any) -> bool:
        if not self.packed_shared or buffer not in (self.plan.lhs_buffer, self.plan.rhs_buffer):
            return False
        if len(values) != 4:
            raise ValueError('packed tensor staging requires four f32 values')
        b = self.lowering.builder
        high, restored = self.bf16(values)
        low, _ = self.bf16([b.fsub(v, h) for v, h in zip(values, restored, strict=True)])
        words = ir.VectorType(I32, 2)
        for part, packed in enumerate((high, low)):
            pointer = self.packed_address(buffer, index, part)
            value = ir.Constant(words, ir.Undefined)
            for i, word in enumerate(packed):
                value = b.insert_element(value, word, integer(i))
            b.store(value, b.bitcast(pointer, words.as_pointer(3)), align=8)
        return True

    def ldmatrix(self, pointer: Any, count: Any, transpose: Any = False) -> Any:
        b = self.lowering.builder
        result = ir.LiteralStructType([I32] * count)
        registers = ','.join('$' + str(i) for i in range(count))
        instruction = 'ldmatrix.sync.aligned.m8n8.x' + str(count) + ('.trans' if transpose else '') + '.shared.b16'
        text = instruction + ' {' + registers + '}, [$' + str(count) + '];'
        signature = ir.FunctionType(result, [I32])
        asm = ir.InlineAsm(signature, text, ','.join(['=r'] * count + ['r', '~{memory}']), side_effect=True)
        value = b.call(asm, [b.ptrtoint(pointer, I32)])
        value.attributes.add('convergent')
        return [b.extract_value(value, i) for i in range(count)]

    def _emit_packed(self) -> None:
        lower, plan = self.lowering, self.plan
        b = lower.builder
        lane, warp = lower.get('lane'), lower.get('warp')
        lhs_base = [lower.expression(v) for v in plan.lhs_base]
        rhs_base = [lower.expression(v) for v in plan.rhs_base]
        row = b.add(lhs_base[0], b.add(b.mul(b.udiv(warp, integer(self.warps_n)), integer(16)), b.urem(lane, integer(16))))
        column = b.add(rhs_base[1], b.mul(b.urem(warp, integer(self.warps_n)), integer(8)))
        accumulators = [b.load(slot) for slot in self.accumulators]
        def fragments(k: int) -> Any:
            lhs_index = [row, b.add(lhs_base[1], b.add(k if isinstance(k, ir.Value) else integer(k), b.mul(b.udiv(lane, integer(16)), integer(8))))]
            rhs_index = [b.add(rhs_base[0], b.add(k if isinstance(k, ir.Value) else integer(k), b.urem(lane, integer(16)))), column]
            lhs_high = self.ldmatrix(self.packed_address(plan.lhs_buffer, lhs_index), 4)
            lhs_low = self.ldmatrix(self.packed_address(plan.lhs_buffer, lhs_index, True), 4)
            rhs_high = self.ldmatrix(self.packed_address(plan.rhs_buffer, rhs_index), 2, True)
            rhs_low = self.ldmatrix(self.packed_address(plan.rhs_buffer, rhs_index, True), 2, True)
            return lhs_high, lhs_low, rhs_high, rhs_low

        current = fragments(0)
        if plan.depth > 16:
            entry = b.block
            body = lower.function.append_basic_block('tensor.prefetch')
            end = lower.function.append_basic_block('tensor.prefetch.end')
            b.branch(body)
            b.position_at_end(body)
            k = b.phi(I32)
            k.add_incoming(integer(16), entry)
            carried = [[b.phi(I32) for _ in part] for part in current]
            for part, values in zip(carried, current, strict=True):
                for phi, value in zip(part, values, strict=True):
                    phi.add_incoming(value, entry)
            sums = [b.phi(F32) for _ in accumulators]
            for phi, value in zip(sums, accumulators, strict=True):
                phi.add_incoming(value, entry)
            following = fragments(k)
            lhs_high, lhs_low, rhs_high, rhs_low = carried
            accumulators = self.mma(lhs_low, rhs_high, sums)
            accumulators = self.mma(lhs_high, rhs_low, accumulators)
            accumulators = self.mma(lhs_high, rhs_high, accumulators)
            next_k = b.add(k, integer(16))
            k.add_incoming(next_k, b.block)
            for part, values in zip(carried, following, strict=True):
                for phi, value in zip(part, values, strict=True):
                    phi.add_incoming(value, b.block)
            for phi, value in zip(sums, accumulators, strict=True):
                phi.add_incoming(value, b.block)
            latch = b.cbranch(b.icmp_unsigned('<', next_k, integer(plan.depth)), body, end)
            metadata: Any = ir.MDValue(lower.module, (), name=str(len(lower.module.metadata)))
            metadata.operands = (metadata, lower.module.add_metadata(['llvm.loop.unroll.disable']))
            latch.set_metadata('llvm.loop', metadata)
            b.position_at_end(end)
            current = following
        lhs_high, lhs_low, rhs_high, rhs_low = current
        accumulators = self.mma(lhs_low, rhs_high, accumulators)
        accumulators = self.mma(lhs_high, rhs_low, accumulators)
        accumulators = self.mma(lhs_high, rhs_high, accumulators)
        for slot, value in zip(self.accumulators, accumulators, strict=True):
            b.store(value, slot)

    def _emit(self) -> None:
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

        def load_lhs(k_offset: Any, row_offset: Any) -> Any:
            if k_offset >= plan.depth:
                return ir.Constant(F32, 0.0)
            indices = [b.add(lhs_base[0], b.add(row, integer(row_offset))), b.add(lhs_base[1], b.add(quad_k, integer(k_offset)))]
            return lower.load(lower.address(plan.lhs_buffer, indices), plan.lhs_buffer, 4)

        def load_lhs_pair(k_offset: Any, row_offset: Any) -> Any:
            if k_offset >= plan.depth:
                return [ir.Constant(F32, 0.0)] * 2
            if known_multiple(plan.lhs_base[1]) % 2:
                return [load_lhs(k_offset + i, row_offset) for i in range(2)]
            indices = [b.add(lhs_base[0], b.add(row, integer(row_offset))), b.add(lhs_base[1], b.add(quad_k, integer(k_offset)))]
            pointer = lower.address(plan.lhs_buffer, indices)
            vector_type = ir.VectorType(F32, 2)
            vector = b.load(b.bitcast(pointer, vector_type.as_pointer(pointer.type.addrspace)), align=4)
            return [b.extract_element(vector, integer(i)) for i in range(2)]

        def load_rhs(k_offset: Any) -> Any:
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

    def emit_store(self, statement: Any) -> bool:
        if not any((statement is candidate for candidate in self.plan.writes)):
            return False
        if statement is not self.plan.writes[0]:
            return True
        self.active(self._store)
        return True

    def _store(self) -> None:
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
