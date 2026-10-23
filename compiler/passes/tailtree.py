
from dataclasses import replace
from typing import Any

from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Load
from compiler.analysis import kind, walk
from compiler.passes.vectorize import _uses_after
from compiler.transforms import default_unroll, striped_reduction

MAX_LENGTH = 8


def _writes_buffer(stmt: Any, buffer: str) -> bool:
    for node in walk((stmt,)):
        tag = kind(node)
        if tag == 'Write' and node.output == buffer:
            return True
        if tag == 'CpAsyncCopy' and node.smem == buffer:
            return True
    return False


def _coalesced_dependent_loop(tile: Any, body: list, at: int, threads: int, unroll: int) -> bool:
    rest = replace(tile, body=Body(tuple(s for k, s in enumerate(body) if k != at)))
    for idx in range(at):
        loop = body[idx]
        if kind(loop) != 'StridedLoop':
            continue
        if not striped_reduction(loop, tile, threads):
            continue
        if striped_reduction(loop, rest, threads):
            continue
        step = loop.step
        extent = loop.axis.extent.value
        if kind(step) != 'Literal' or not isinstance(step.value, int) or step.value <= 0:
            return True
        if extent <= 0 or extent % step.value:
            return True
        cap = unroll or default_unroll(loop, None, set())
        if extent // step.value > cap:
            return True
    return False


def _linearize_in_tile(tile: Any, tag: int, threads: int, unroll: int) -> Any:
    smems = [s for s in walk(tile.body) if kind(s) == 'Smem']
    body = list(tile.body)
    for at, stmt in enumerate(body):
        if kind(stmt) != 'TreeHalve':
            continue
        if stmt.length < 2 or stmt.length > MAX_LENGTH or stmt.length & stmt.length - 1:
            continue
        if stmt.inner is not None:
            continue
        if len(stmt.bufs) != 1 or len(stmt.state) != 1 or len(stmt.state_b) != 1:
            continue
        if stmt.dtype is not None and getattr(stmt.dtype, 'name', None) != 'f32':
            continue
        if len(stmt.combine_states) != 1:
            continue
        combine = stmt.combine_states[0]
        if kind(combine) != 'Assign' or combine.name != stmt.state[0]:
            continue
        if set(combine.args) != {stmt.state[0], stmt.state_b[0]}:
            continue
        if combine.op.name != 'add':
            continue
        buf = stmt.bufs[0]
        decls = [s for s in smems if s.name == buf]
        if len(decls) != 1:
            continue
        decl = decls[0]
        if decl.dtype != 'float' or len(decl.extents) != 1:
            continue
        size = 1
        for extent in decl.extents:
            size *= int(extent)
        if size < stmt.length:
            continue
        if len(smems) != 1:
            continue
        if _coalesced_dependent_loop(tile, body, at, threads, unroll):
            continue
        after = _uses_after(tile, at)
        if after is None or stmt.state_b[0] in after:
            continue
        last_write = None
        for idx in range(at):
            if _writes_buffer(body[idx], buf):
                last_write = idx
        if last_write is None:
            continue
        if not any((kind(s) == 'Sync' and not s.warp for s in body[last_write + 1:at])):
            continue
        written_before: set[str] = set()
        for idx in range(at + 1):
            for node in walk((body[idx],)):
                if kind(node) == 'Write':
                    written_before.add(node.output)
        read_after: set[str] = set()
        for idx in range(at + 1, len(body)):
            for node in walk((body[idx],)):
                if kind(node) == 'Load':
                    read_after.add(node.input)
        if (written_before & read_after) - {buf}:
            continue
        bound = set()
        for node in walk(tile.body):
            name = kind(node)
            if name == 'Load':
                bound.update(node.names)
            elif name in ('Assign', 'Accum', 'Init', 'Let'):
                bound.add(node.name)
            elif name in ('WarpShuffle', 'TreeHalve'):
                bound.update(node.state)
                bound.update(node.state_b)
            elif name in ('Loop', 'StridedLoop'):
                bound.add(node.axis.name)
            elif name == 'Smem':
                bound.add(node.name)
            elif name == 'Select':
                bound.add(node.name)
        lanes = [f'__tail_{tag}_{at}_{m}' for m in range(stmt.length)]
        if set(lanes) & bound:
            continue
        level = list(lanes)
        stmts: list[Any] = [
            Load(names=(name,), input=buf, index=(Literal(m, dtype='int'),), dtype=getattr(combine, 'dtype', None))
            for m, name in enumerate(lanes)
        ]
        depth = 0
        ok = True
        while len(level) > 1:
            nxt = []
            for p in range(0, len(level), 2):
                a, b = level[p], level[p + 1]
                if len(level) == 2:
                    name = stmt.state[0]
                else:
                    name = f'__tail_{tag}_{at}_p{depth}_{p // 2}'
                    if name in bound:
                        ok = False
                        break
                    bound.add(name)
                stmts.append(replace(combine, name=name, args=(a, b)))
                nxt.append(name)
            if not ok:
                break
            level = nxt
            depth += 1
        if not ok or len(level) != 1 or level[0] != stmt.state[0]:
            continue
        merged = [*body[:at], *stmts, *body[at + 1:]]
        return replace(tile, body=Body(tuple(merged)))
    return None


def linearize_small_trees(kernel: Any, launch: Any, options: Any = None) -> Any:
    threads = launch['block'][0]
    unroll = options.unroll if options is not None else 0
    changed = False
    tiles = []
    for tag, stmt in enumerate(kernel.body):
        if kind(stmt) != 'Tile':
            tiles.append(stmt)
            continue
        tile = _linearize_in_tile(stmt, tag, threads, unroll)
        if tile is None:
            tiles.append(stmt)
        else:
            tiles.append(tile)
            changed = True
    if not changed:
        return None
    return replace(kernel, body=Body(tuple(tiles)))
