import math
from typing import Any
from compiler.passes.regstage import _evaluate

def recurrence_storage_is_warp_private(plan: Any, launch: Any, buffer: Any, shape: Any, count: Any) -> bool:
    try:
        threads, blocks = launch['block'][0], launch['grid'][0]
        tile = plan.tile
        parts = len(plan.reductions)
        depth = plan.reductions[0].axis.extent.value
        names = [axis.name for axis in tile.axes]
        counters = [plan.loop.axis.name, plan.reductions[0].axis.name]
        extents = [axis.extent.value for axis in tile.axes]
        if not (math.isfinite(plan.identity) and plan.identity < 0) or plan.width != 32 or tile.raster_group or tile.aux_threads or type(threads) is not int or type(blocks) is not int or not 0 < threads <= 1024 or threads % 32 or not 0 < threads * blocks < 2**32:
            return False
        if any(type(n) is not int or n <= 0 for n in (*shape, *extents, depth)) or math.prod(extents) != threads * blocks or math.prod(shape) >= 2**31:
            return False
        if len(set(names + counters)) != len(names + counters) or any(n in ('lane', 'warp') or n.startswith(('threadIdx.', 'blockIdx.', 'blockDim.', 'gridDim.')) for n in names + counters):
            return False
        readers = [(part, load) for part, load in enumerate(plan.invariant) if load.input == buffer]
        if not readers or any(load.input == buffer for load in (*plan.varying, *plan.gather)) or any(len(load.names) != 1 or len(load.index) != len(shape) for _, load in readers):
            return False
        if blocks * (threads // 32) * depth * len(readers) > 65536 or count != (threads // 32) * parts * plan.extent:
            return False
        for block in range(blocks):
            for warp in range(threads // 32):
                ticket = block * threads + warp * 32
                values = {}
                for name, extent in reversed(list(zip(names, extents, strict=True))):
                    values[name], ticket = ticket % extent, ticket // extent
                for dot in range(depth):
                    values[counters[1]] = dot
                    for part, load in readers:
                        address = 0
                        for expression, extent in zip(load.index, shape, strict=True):
                            coordinate = _evaluate(expression, values)
                            if not 0 <= coordinate < extent:
                                return False
                            address = address * extent + coordinate
                        if address < count and (address // (parts * plan.extent) != warp or address // plan.extent % parts != part):
                            return False
        return True
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
