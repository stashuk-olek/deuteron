import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import compile_kernel
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import Tile
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.leaves import Assign, Load, Write


def example(kind='affine', count=512, width=4, offset=0):
    if kind not in ('copy', 'widen', 'narrow', 'affine', 'cancellation') or width not in (1, 2, 4) or count <= 0 or count % width:
        raise ValueError('unsupported example configuration')
    source = F32 if kind == 'narrow' else F16
    destination = F32 if kind == 'widen' else F16
    index = Var('element') * width
    names = tuple(f'input_{i}' for i in range(width))
    body = [Load(names=names, input='x', index=(index + offset,), dtype=source)]
    args = [{'name': 'x', 'dtype': source.name, 'shape': [count + offset], 'seed': 7}]
    values = names
    value_dtype = source
    if kind == 'affine':
        args += [{'name': 'scale', 'dtype': 'f32', 'shape': [1], 'data': [1.5]}, {'name': 'bias', 'dtype': 'f32', 'shape': [1], 'data': [-0.25]}]
        body += [Load(name='scale_value', input='scale', index=(Literal(0, dtype='int'),), dtype=F32), Load(name='bias_value', input='bias', index=(Literal(0, dtype='int'),), dtype=F32)]
        values = tuple(f'result_{i}' for i in range(width))
        for i, name in enumerate(names):
            body += [Assign(f'product_{i}', 'multiply', (name, 'scale_value'), F32), Assign(values[i], 'add', (f'product_{i}', 'bias_value'), F32)]
        value_dtype = F32
    elif kind == 'cancellation':
        args.append({'name': 'y', 'dtype': 'f16', 'shape': [count], 'seed': 8})
        second = tuple(f'other_{i}' for i in range(width))
        body.append(Load(names=second, input='y', index=(index,), dtype=F16))
        values = tuple(f'result_{i}' for i in range(width))
        for i, name in enumerate(names):
            body += [Assign(f'sum_{i}', 'add', (name, second[i]), F32), Assign(values[i], 'subtract', (f'sum_{i}', name), F32)]
        value_dtype = F32
    elif kind == 'widen':
        values = tuple(f'result_{i}' for i in range(width))
        body += [Assign(value, 'copy', (name,), F32) for value, name in zip(values, names)]
        value_dtype = F32
    body.append(Write(output='result', index=(index,), values=values, value_dtype=value_dtype))
    args.append({'name': 'result', 'dtype': destination.name, 'shape': [count], 'output': True})
    tile = Tile(axes=(Axis(name='element', extent=Dim(count // width)),), body=Body(body), block_threads=128)
    entry = 'f16_storage_' + kind
    inputs = [a['name'] for a in args if not a.get('output')]
    output = {'name': 'result', 'shape': [count], 'dtype': destination.name}
    nodes = {a['name']: {'op': 'InputOp', 'op_fields': {}, 'inputs': [], 'outputs': [{'name': a['name'], 'shape': a['shape'], 'dtype': a['dtype']}]} for a in args if not a.get('output')}
    nodes['result'] = {'op': 'KernelOp', 'op_fields': {'body': [repr(tile)], 'name': entry, 'zero_delegated': [], 'serial': []}, 'inputs': inputs, 'outputs': [output]}
    graph = {'inputs': inputs, 'outputs': ['result'], 'nodes': nodes}
    launch = {'node': 'result', 'entry': entry, 'arg_order': [a['name'] for a in args], 'grid': [math.ceil(count / width / 128), 1, 1], 'block': [128, 1, 1], 'smem_bytes': 0, 'zero_outputs': [], 'args': args, 'final': True}
    metadata = {'id': entry, 'rtol': 0.001, 'atol': 0.00001, 'graph_inputs': inputs, 'graph_outputs': ['result'], 'kernels': [launch]}
    return graph, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    from benchmark import LocalRunner, argument_array, compare_outputs
    runner = LocalRunner(args.device) if args.device is not None else None
    try:
        for name in ('copy', 'widen', 'narrow', 'affine', 'cancellation'):
            graph, meta = example(name)
            folder = args.output / name
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / 'kernel.ir.json'
            source.write_text(json.dumps(graph, indent=2))
            (folder / 'meta.json').write_text(json.dumps(meta, indent=2))
            launch = meta['kernels'][0]
            inputs = {a['name']: argument_array(a) for a in launch['args'] if not a.get('output')}
            result = inputs['x'].astype(np.float32)
            if name == 'affine':
                result = result * inputs['scale'][0] + inputs['bias'][0]
            elif name == 'cancellation':
                result = (result + inputs['y'].astype(np.float32)) - result
            result = result.astype(np.float32 if name == 'widen' else np.float16)
            np.savez(folder / 'expected.npz', result=result)
            compiled = compile_kernel(source, opt_level=3)
            (folder / 'kernel.ptx').write_text(compiled.ptx)
            if runner is not None:
                actual = runner.run_kernel(launch, compiled.ptx, warmup=0, iters=1, samples=1)['outputs']
                compare_outputs(folder, meta, actual)
            print(name, 'PASS' if runner is not None else 'compiled', folder)
    finally:
        if runner is not None:
            runner.close()


if __name__ == '__main__':
    main()
