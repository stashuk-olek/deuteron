import math
from compiler.analysis import free_variables, index_variables, integer, kind
SCALAR_OPS = {'add', 'subtract', 'multiply', 'divide', 'copy', 'reciprocal', 'exp', 'exp_fast', 'sqrt', 'rsqrt', 'maximum', 'minimum', 'relu', 'negative', 'absolute'}

def reciprocal_sqrt_source(statement, previous):
    if type(statement).__name__ == 'Assign' and statement.op.name == 'reciprocal' and (type(previous).__name__ == 'Assign') and (previous.op.name == 'sqrt') and (len(previous.args) == 1) and (statement.args == (previous.name,)) and (previous.name != previous.args[0]):
        return previous.args[0]
    return None

def sharing_width(loop, tile, uniform_axes):
    if type(loop).__name__ != 'Loop' or len(loop.body) != 4:
        return 1
    if (tile.block_threads or 256) % 32:
        return 1
    if tile.raster_group and tile.raster_axes:
        for index, axis in enumerate(tile.axes):
            if axis.name in tile.raster_axes and axis.extent.value > 1:
                stride = math.prod((a.extent.value for a in tile.axes[index + 1:]))
                if stride % 32:
                    return 1
    left, right, product, accum = loop.body
    if tuple((type(s).__name__ for s in loop.body)) != ('Load', 'Load', 'Assign', 'Accum'):
        return 1
    if len(left.names) != 1 or len(right.names) != 1 or not product.op.semiring_product:
        return 1
    if set(product.args) != {left.names[0], right.names[0]}:
        return 1
    if not (accum.op.associative and accum.op.identity == 0) or not product.op.distributes_over(accum.op) or accum.base is not None or accum.value != product.name:
        return 1
    used = set()
    for load in (left, right):
        names = index_variables(load.index)
        if names is None:
            return 1
        used.update(names)
    used.discard(loop.axis.name)
    if used - {axis.name for axis in tile.axes} - uniform_axes:
        return 1
    width = 1
    for axis in reversed(tile.axes):
        if axis.name in used and axis.extent.value != 1:
            break
        width *= axis.extent.value
        if width >= 32:
            if width % 32:
                return 1
            width = 32
            break
    if width & width - 1 or loop.axis.extent.value < width:
        return 1
    return width

def plain_reduction(loop):
    if type(loop).__name__ != 'StridedLoop':
        return False
    if any((type(s).__name__ not in ('Load', 'Assign', 'Accum', 'Let') for s in loop.body)):
        return False
    accumulators = [s for s in loop.body if type(s).__name__ == 'Accum']
    return len(accumulators) == 1 and accumulators[0].op.associative and accumulators[0].op.identity == 0 and (accumulators[0].base is None)

def striped_reduction(loop, tile, threads):
    if not plain_reduction(loop) or len(loop.body) not in (2, 3) or (not loop.seed):
        return False
    load, accum = (loop.body[0], loop.body[-1])
    if type(load).__name__ != 'Load' or len(load.names) != 1:
        return False
    if len(loop.body) == 3:
        square = loop.body[1]
        if type(square).__name__ != 'Assign' or not square.op.semiring_product:
            return False
        if square.args != (load.names[0], load.names[0]) or accum.value != square.name:
            return False
    elif accum.value != load.names[0]:
        return False
    if type(loop.start).__name__ != 'Var' or loop.start.name != tile.axes[-1].name:
        return False
    if type(loop.step).__name__ != 'Literal' or loop.step.value != threads or tile.axes[-1].extent.value != threads:
        return False
    if loop.end is not None or loop.axis.extent.value % (threads * 4):
        return False
    if not load.index or type(load.index[-1]).__name__ != 'Var' or load.index[-1].name != loop.axis.name:
        return False
    row_axes = {axis.name for axis in tile.axes[:-1]}
    for index in load.index[:-1]:
        names = index_variables((index,))
        if names is None or names - row_axes:
            return False
    for index, child in enumerate(tile.body[:-1]):
        if child is loop:
            combine = tile.body[index + 1]
            if type(combine).__name__ != 'WarpShuffle' or combine.state != (accum.name,) or combine.length != 32:
                return False
            return any((type(s).__name__ == 'TreeHalve' and s.state == (accum.name,) and (s.length == threads // 32) and (s.tid_var == 'warp') for s in tile.body[index + 2:]))
    return False

def default_unroll(loop, tile, uniform_axes):
    if plain_reduction(loop):
        return 32
    if tile is not None:
        inner = sum((sharing_width(child, tile, uniform_axes | {loop.axis.name}) > 1 for child in loop.body))
        if inner == 1:
            return 32
    return 8

def shared_layout(columns, role, mode):
    if role == 'row':
        shift = 3 if mode == 'bf16x3' else 2
        return (shift, min(3 if mode == 'bf16x3' else 7, columns - 1 >> shift), False)
    return (3, min(3, columns - 1 >> 3), mode == 'bf16x3')

def symbolic(op, *args):
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

def uniform_prefix_values(prefix, tile, width):
    uniform_axes = {'warp'}
    for index, axis in enumerate(tile.axes):
        stride = math.prod((a.extent.value for a in tile.axes[index + 1:]))
        if axis.extent.value == 1 or stride % width == 0:
            uniform_axes.add(axis.name)
    values = {}
    for statement in prefix:
        tag = kind(statement)
        if tag == 'Load':
            names = index_variables(statement.index)
            if names is None:
                return None
            uniform = not names - uniform_axes
            values.update(((name, uniform) for name in statement.names))
        elif tag == 'Let':
            names = free_variables(statement.value)
            if names is None:
                return None
            known_uniform = uniform_axes | {name for name, uniform in values.items() if uniform}
            values[statement.name] = not names - known_uniform
        elif tag == 'Assign':
            if statement.op.name not in SCALAR_OPS:
                return None
            values[statement.name] = all((values.get(name, False) for name in statement.args))
        else:
            return None
    return values

