import math
from dataclasses import replace
from typing import Any

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import WarpShuffle
from emmy.compiler.ir.stmt.base import Stmt
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Load
from compiler.analysis import kind, walk
from compiler.passes.regforward import _bound_names
from compiler.passes.vectorize import _uses_after

MAX_LENGTH = 8


def _accesses(body: Any) -> tuple[set[str], set[str]]:
    reads, writes = set(), set()
    for stmt in walk(body):
        tag = kind(stmt)
        if tag == 'Load':
            reads.add(stmt.input)
        elif tag == 'Write':
            writes.add(stmt.output)
        elif tag == 'CpAsyncCopy':
            reads.add(stmt.src)
            writes.add(stmt.smem)
        elif tag == 'TreeHalve':
            reads.update(stmt.bufs)
            writes.update(stmt.bufs)
    return reads, writes


def _scalar_tree(stmt: Any, tag: int, bound: set) -> Any:
    lanes = [f'__tail_0_{tag}_{m}' for m in range(stmt.length)]
    if set(lanes) & bound:
        return None
    combine = stmt.combine_states[0]
    stmts: list[Any] = [Load(names=(name,), input=stmt.bufs[0], index=(Literal(m, dtype='int'),), dtype=combine.dtype) for m, name in enumerate(lanes)]
    depth = 0
    while len(lanes) > 1:
        level = []
        for p in range(0, len(lanes), 2):
            name = stmt.state[0] if len(lanes) == 2 else f'__tail_0_{tag}_p{depth}_{p // 2}'
            if len(lanes) != 2 and name in bound:
                return None
            bound.add(name)
            stmts.append(replace(combine, name=name, args=(lanes[p], lanes[p + 1])))
            level.append(name)
        lanes = level
        depth += 1
    return stmts


def linearize_small_trees(kernel: Any, launch: Any, options: Any) -> Any:
    if len(kernel.body) != 1 or kind(kernel.body[0]) != 'Tile':
        return None
    tile = kernel.body[0]
    threads = launch['block'][0]
    if threads % 32 or not tile.axes or tile.axes[-1].extent.value != threads:
        return None
    if math.prod(a.extent.value for a in tile.axes) != threads * launch['grid'][0]:
        return None
    smems = [s for s in walk(tile.body) if kind(s) == 'Smem']
    body = list(tile.body)
    for at, stmt in enumerate(body):
        if kind(stmt) != 'TreeHalve' or stmt.tid_var != 'warp' or stmt.inner is not None or stmt.barrier_id:
            continue
        if not 2 <= stmt.length <= MAX_LENGTH or stmt.length & (stmt.length - 1):
            continue
        bufs, states, others = set(stmt.bufs), set(stmt.state), set(stmt.state_b)
        if not len(bufs) == len(states) == len(others) == len(stmt.bufs) == len(stmt.state) == len(stmt.state_b):
            continue
        if not bufs or states & others or stmt.dtype is not None and stmt.dtype.name != 'f32':
            continue
        if len(smems) != len(bufs) or {s.name for s in smems} != bufs:
            continue
        if any(s.dtype != 'float' or len(s.extents) != 1 or s.extents[0] < stmt.length for s in smems):
            continue
        bound = states | others
        defined = set()
        for combine in stmt.combine_states:
            if kind(combine) != 'Assign' or set(combine.args) - bound or combine.dtype is not None and combine.dtype.name != 'f32':
                break
            bound.add(combine.name)
            defined.add(combine.name)
        else:
            after = _uses_after(tile, at)
            if not states <= defined or after is None or ((defined | others) - states) & after:
                continue
            before_reads, before_writes = _accesses(body[:at])
            after_reads, after_writes = _accesses(body[at + 1:])
            if not bufs <= before_writes or bufs & (after_reads | after_writes):
                continue
            if before_writes & (after_reads | after_writes) or before_reads & after_writes:
                continue
            last_write = max(i for i, s in enumerate(body[:at]) if _accesses((s,))[1] & bufs)
            if not any(kind(s) == 'Sync' and not s.warp and not s.barrier_id for s in body[last_write + 1:at]):
                continue
            scalar = len(states) == len(stmt.combine_states) == 1 and stmt.combine_states[0].op.name == 'add' and set(stmt.combine_states[0].args) == states | others
            stmts: list[Stmt]
            if scalar:
                stmts = _scalar_tree(stmt, at, _bound_names(tile))
                if stmts is None:
                    continue
            elif options.tree_specialization:
                slot = BinaryExpr(op='&', left=Var('lane'), right=Literal(stmt.length - 1, dtype='int'))
                stmts = [Load(names=(state,), input=buf, index=(slot,), dtype=stmt.dtype) for buf, state in zip(stmt.bufs, stmt.state)]
                stmts.append(WarpShuffle(state=stmt.state, state_b=stmt.state_b, combine_states=stmt.combine_states, length=stmt.length, dtype=stmt.dtype))
            else:
                continue
            tile = replace(tile, body=Body((*body[:at], *stmts, *body[at + 1:])))
            return replace(kernel, body=Body((tile,)))
    return None
