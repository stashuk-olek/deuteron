import os

for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, '1')

from compiler.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
