import argparse
import os
import sys
from pathlib import Path
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')
from compiler import Options, compile_kernel

def main():
    parser = argparse.ArgumentParser(description='Compile Emmy Kernel IR to PTX through LLVM')
    parser.add_argument('kernel', type=Path)
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--opt-level', type=int, choices=range(4), default=3)
    parser.add_argument('--unroll', type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=0)
    parser.add_argument('--vector-width', type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument('--contraction-precision', choices=('f32', 'tf32x3', 'bf16x3'), default='bf16x3')
    for flag in ('approximate-math', 'reduction-sharing', 'recurrence-staging', 'tree-specialization', 'coalesced-reductions', 'warp-willreturn', 'elementwise-vectorize'):
        parser.add_argument('--' + flag, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dump-llvm', type=Path)
    parser.add_argument('--dump-unoptimized', type=Path)
    args = parser.parse_args()
    try:
        options = Options(**{name: getattr(args, name) for name in Options.__dataclass_fields__})
        result = compile_kernel(args.kernel, arch=args.arch, opt_level=args.opt_level, options=options)
        for path, content in ((args.dump_llvm, result.optimized), (args.dump_unoptimized, result.unoptimized)):
            if path:
                path.write_text(content)
        sys.stdout.write(result.ptx)
    except (ValueError, NotImplementedError, RuntimeError) as exc:
        parser.exit(1, f'compile: {exc}\n')
if __name__ == '__main__':
    main()
