import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triton_configs(name, args, kernels, triton):
    if name.startswith(('p00', 'p01', 'p03')):
        fn = kernels.relu_product_kernel if name.startswith('p03') else kernels.add_kernel
        n = math.prod(next(iter(args.values()))['shape'])
        return fn, [(dict(N=n, BLOCK=1024), [triton.cdiv(n, 1024), 1, 1], 4, 2)]
    if name.startswith('p02'):
        n = math.prod(args['x']['shape'])
        return kernels.affine_kernel, [(dict(N=n, BLOCK=1024), [triton.cdiv(n, 1024), 1, 1], 4, 2)]
    if name.startswith(('r01', 'r02', 'r03')):
        fn = kernels.mean_kernel if name.startswith('r03') else kernels.sum_kernel
        n = math.prod(args['x']['shape'])
        return fn, [(dict(N=n, BLOCK=min(n, 4096)), [1, 1, 1], 4, 2)]
    if name.startswith(('r04', 'f01', 'f02', 'f03')):
        fn = {'r04': kernels.row_sum_kernel, 'f01': kernels.rms_kernel, 'f02': kernels.softmax_kernel, 'f03': kernels.l2_kernel}[name[:3]]
        rows, cols = args['x']['shape']
        return fn, [(dict(COLS=cols, BLOCK=triton.next_power_of_2(cols)), [rows, 1, 1], 4, 2)]
    if name.startswith('m'):
        m, k = args['x0']['shape']
        _, n = args['x1']['shape']
        return kernels.matmul_kernel, [(dict(M=m, N=n, K=k, BM=bm, BN=64, BK=bk), [triton.cdiv(m, bm) * triton.cdiv(n, 64), 1, 1], 4, 3) for bm, bk in ((32, 32), (64, 32), (32, 64))]
    if name.startswith('a'):
        m, d = args['q']['shape']
        _, n = args['kt']['shape']
        return kernels.attention_kernel, [(dict(M=m, N=n, D=d, BM=bm), [triton.cdiv(m, bm), 1, 1], warps, 2) for bm, warps in ((16, 4), (32, 4), (16, 8))]
    raise ValueError(f'No comparison kernel for {name}')


def main():
    parser = argparse.ArgumentParser(description='Compile the 15 open workloads for the RTX 5090 comparison without using a GPU')
    parser.add_argument('compiler', choices=('deuteron', 'nvcc', 'triton'))
    parser.add_argument('--problems-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvcc', default='nvcc')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        raise FileExistsError(f'Use a new output directory: {manifest}')
    if args.compiler == 'triton':
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        import triton_kernels as kernels
        label = 'Triton' + triton.__version__
    elif args.compiler == 'deuteron':
        sys.path.insert(0, str(root))
        from compiler import compile_kernel
        import llvmlite
        from llvmlite import binding as llvm
        label = 'Deuteron'
    else:
        version = subprocess.check_output([args.nvcc, '--version'], text=True)
        label = 'NVCC' + re.search(r'V([\d.]+)', version).group(1)
    records = []
    for problem in sorted(args.problems_dir.iterdir()):
        if not (problem / 'meta.json').is_file():
            continue
        meta = json.loads((problem / 'meta.json').read_text())
        if len(meta['kernels']) != 1:
            raise ValueError(f'Expected one kernel: {problem}')
        original = meta['kernels'][0]
        if args.compiler == 'triton':
            if any(a['dtype'] != 'f32' or a.get('scalar') for a in original['args']):
                raise ValueError('The Triton comparison kernels require f32 buffer arguments')
            fn, configs = triton_configs(problem.name, {a['name']: a for a in original['args']}, kernels, triton)
        else:
            configs = [None]
        for config, settings in enumerate(configs):
            dest = args.output / f'{problem.name}-{config}'
            dest.mkdir()
            launch = copy.deepcopy(original)
            record = {'problem': problem.name, 'compiler': label, 'config': config, 'meta_sha256': sha(problem / 'meta.json'), 'expected_sha256': sha(problem / 'expected.npz')}
            started = time.perf_counter()
            if args.compiler == 'triton':
                constants, grid, warps, stages = settings
                count = len(original['arg_order'])
                signature = {n: '*fp32' for n in fn.arg_names[:count]}
                attrs = {(i,): [('tt.divisibility', 16)] for i in range(count)}
                options = {'num_warps': warps, 'num_stages': stages, 'enable_fp_fusion': True, 'ptx_version': 87}
                compiled = triton.compile(ASTSource(fn, signature, constants, attrs), target=GPUTarget('cuda', 120, 32), options=options)
                if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
                    raise ValueError('This harness only supports zero-sized Triton scratch buffers')
                for suffix in ('ptx', 'llir', 'ttir', 'ttgir'):
                    (dest / ('kernel.' + suffix)).write_text(compiled.asm[suffix])
                launch.update(entry=compiled.metadata.name, grid=grid, block=[warps * 32, 1, 1], smem_bytes=compiled.metadata.shared, zero_outputs=[])
                for name in ('__triton_global_scratch', '__triton_profile_scratch'):
                    launch['arg_order'].append(name)
                    launch['args'].append({'name': name, 'dtype': 'f32', 'shape': [1], 'data': [0.0]})
                record.update(options=options, constants=constants, target='sm_120a', source_sha256=sha(Path(kernels.__file__)))
            elif args.compiler == 'deuteron':
                source = problem / (original['entry'] + '.ir.json')
                compiled = compile_kernel(source, arch='sm_120', opt_level=3)
                (dest / 'kernel.ptx').write_text(compiled.ptx)
                (dest / 'kernel.ll').write_text(compiled.optimized)
                (dest / 'kernel.unoptimized.ll').write_text(compiled.unoptimized)
                record.update(target='sm_120', opt_level=3, source_sha256=sha(source), compiler_files={str(p.relative_to(root)): sha(p) for p in sorted((root / 'compiler').rglob('*.py'))}, llvmlite=llvmlite.__version__, llvm=llvm.llvm_version_info)
            else:
                source = problem / (original['entry'] + '.ref.cu')
                command = [args.nvcc, '-O3', '--std=c++17', '-arch=compute_120', '--ptx', str(source), '-o', str(dest / 'kernel.ptx')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                record.update(target='compute_120', command=command, source_sha256=sha(source), version=version)
            record.update(launch=launch, ptx=str((dest / 'kernel.ptx').relative_to(args.output)), ptx_sha256=sha(dest / 'kernel.ptx'), compile_seconds=time.perf_counter() - started)
            records.append(record)
            manifest.write_text(json.dumps(records, indent=2) + '\n')
            print(label, problem.name, config, 'compiled', flush=True)


if __name__ == '__main__':
    main()
