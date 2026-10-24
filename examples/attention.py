import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import compile_kernel
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write


def example(rows=3, keys=64, depth=32, inplace_denominator=False, seed=7):
    if any(type(value) is not int or value <= 0 for value in (rows, keys, depth)) or keys % 32 or keys > 256 or depth < 32:
        raise ValueError('use positive rows, 32..256 keys in multiples of 32, and depth >= 32')
    op = ElementwiseImpl

    def assign(name, operation, *args):
        return Assign(name, op(operation), args, dtype=F32)

    dot = Loop(Axis('dot', Dim(depth)), Body((
        Load('query', 'q', (Var('row'), Var('dot')), dtype=F32),
        Load('key_value', 'kt', (Var('dot'), Var('key')), dtype=F32),
        assign('product', 'multiply', 'query', 'key_value'),
        Accum('score_sum', 'product', op('add'), F32, ('dot',)),
    )))
    outer = Loop(Axis('key', Dim(keys)), Body((
        dot,
        assign('score', 'multiply', 'score_sum', 'scale'),
        assign('next_maximum', 'maximum', 'maximum', 'score'),
        assign('old_delta', 'subtract', 'maximum', 'next_maximum'),
        assign('alpha', 'exp', 'old_delta'),
        assign('new_delta', 'subtract', 'score', 'next_maximum'),
        assign('beta', 'exp', 'new_delta'),
        assign('mass_carry', 'multiply', 'mass', 'alpha'),
        Accum('mass', 'beta', op('add'), F32, ('key',), 'mass_carry'),
        assign('weighted_carry', 'multiply', 'weighted', 'alpha'),
        Load('value', 'v', (Var('key'), Var('column')), dtype=F32),
        assign('weighted_new', 'multiply', 'value', 'beta'),
        Accum('weighted', 'weighted_new', op('add'), F32, ('key',), 'weighted_carry'),
        Accum('maximum', 'score', op('maximum'), F32, ('key',)),
    )))
    inverse = 'mass' if inplace_denominator else 'inverse'
    tile = Tile((Axis('row', Dim(rows)), Axis('column', Dim(32))), Body((
        Let('scale', Literal(float(np.float32(1 / math.sqrt(depth))), 'float'), F32),
        outer,
        assign(inverse, 'reciprocal', 'mass'),
        assign('normalized', 'multiply', 'weighted', inverse),
        Write('out', (Var('row'), Var('column')), 'normalized', value_dtype=F32),
    )), block_threads=32)
    shapes = {'q': [rows, depth], 'kt': [depth, keys], 'v': [keys, 32], 'out': [rows, 32]}
    nodes = {name: {'op': 'InputOp', 'op_fields': {}, 'inputs': [], 'outputs': [{'name': name, 'shape': shape, 'dtype': 'f32'}]} for name, shape in shapes.items() if name != 'out'}
    entry = 'attention_example'
    nodes['out'] = {'op': 'KernelOp', 'op_fields': {'body': [repr(tile)], 'name': entry, 'zero_delegated': [], 'serial': []}, 'inputs': ['q', 'kt', 'v'], 'outputs': [{'name': 'out', 'shape': shapes['out'], 'dtype': 'f32'}]}
    graph = {'inputs': ['q', 'kt', 'v'], 'outputs': ['out'], 'nodes': nodes}
    args = [{'name': name, 'dtype': 'f32', 'shape': shapes[name], 'seed': seed + i} for i, name in enumerate(('q', 'kt', 'v'))]
    args.append({'name': 'out', 'dtype': 'f32', 'shape': shapes['out'], 'output': True})
    launch = {'node': 'out', 'entry': entry, 'arg_order': ['q', 'kt', 'v', 'out'], 'grid': [rows, 1, 1], 'block': [32, 1, 1], 'smem_bytes': 0, 'zero_outputs': [], 'args': args, 'final': True}
    metadata = {'id': entry, 'rtol': 0.001, 'atol': 0.0001, 'graph_inputs': graph['inputs'], 'graph_outputs': graph['outputs'], 'kernels': [launch]}
    return graph, metadata


def reference(metadata):
    inputs = {arg['name']: np.random.default_rng(arg['seed']).standard_normal(arg['shape']).astype(np.float32).astype(np.float64) for arg in metadata['kernels'][0]['args'] if not arg.get('output')}
    scores = inputs['q'] @ inputs['kt'] * np.float32(1 / math.sqrt(inputs['q'].shape[1]))
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ inputs['v']


def write_example(folder, **configuration):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    graph, metadata = example(**configuration)
    path = folder / 'kernel.ir.json'
    path.write_text(json.dumps(graph, indent=2) + '\n')
    path.with_name('meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez(folder / 'expected.npz', out=reference(metadata))
    return path, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--keys', type=int, default=64)
    parser.add_argument('--depth', type=int, default=32)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--inplace-denominator', action='store_true')
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    path, metadata = write_example(args.output, rows=args.rows, keys=args.keys, depth=args.depth, inplace_denominator=args.inplace_denominator, seed=args.seed)
    compiled = compile_kernel(path, arch=args.arch, opt_level=3)
    path.with_name('kernel.ptx').write_text(compiled.ptx)
    if args.device is not None:
        from benchmark import LocalRunner, compare_outputs
        with LocalRunner(args.device) as runner:
            result = runner.run_kernel(metadata['kernels'][0], compiled.ptx, warmup=0, iters=1, samples=1)
        compare_outputs(args.output, metadata, result['outputs'])
    print('PASS' if args.device is not None else 'compiled', args.output)


if __name__ == '__main__':
    main()
