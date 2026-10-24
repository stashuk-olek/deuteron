import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import tempfile

from compiler import Options, compile_kernel
from compiler._version import __version__


def _same_file(left: Path, right: Path) -> bool:
    return left == right or (left.exists() and right.exists() and left.samefile(right))


def _dump_paths(kernel: Path, requested: tuple[Path | None, ...]) -> list[Path | None]:
    protected = [kernel.resolve(), kernel.with_name('meta.json').resolve()]
    paths: list[Path | None] = []
    for requested_path in requested:
        if requested_path is None:
            paths.append(None)
            continue
        path = requested_path.resolve()
        if any(_same_file(path, other) for other in protected):
            raise ValueError(f'dump path would overwrite an input file: {requested_path}')
        if any(other is not None and _same_file(path, other) for other in paths):
            raise ValueError('LLVM dump paths must refer to different files')
        if path.exists() and not path.is_file():
            raise ValueError(f'dump path is not a regular file: {requested_path}')
        if not path.parent.is_dir():
            raise ValueError(f'dump parent is not an existing directory: {requested_path.parent}')
        paths.append(path)
    return paths


def _write_dumps(dumps: list[tuple[Path, str]]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in dumps:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                             dir=destination.parent, prefix='.' + destination.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, destination))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _write_ptx(ptx: str) -> None:
    try:
        sys.stdout.write(ptx)
        sys.stdout.flush()
    except OSError:
        try:
            with open(os.devnull, 'w') as sink:
                os.dup2(sink.fileno(), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compile Emmy Kernel IR to PTX through LLVM')
    parser.add_argument('--version', action='version', version=f'deuteron {__version__}')
    parser.add_argument('kernel', type=Path)
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--opt-level', type=int, choices=range(4), default=3)
    parser.add_argument('--unroll', type=int, choices=(0, 1, 2, 4, 8, 16, 32), default=0)
    parser.add_argument('--vector-width', type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument('--contraction-precision', choices=('f32', 'tf32x3', 'bf16x3'), default='bf16x3')
    for flag in ('approximate-math', 'reduction-sharing', 'recurrence-staging', 'tree-specialization',
                 'coalesced-reductions', 'warp-willreturn', 'elementwise-vectorize',
                 'register-forwarding', 'tail-linearize'):
        parser.add_argument('--' + flag, action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dump-llvm', type=Path)
    parser.add_argument('--dump-unoptimized', type=Path)
    args = parser.parse_args(argv)
    try:
        paths = _dump_paths(args.kernel, (args.dump_llvm, args.dump_unoptimized))
        options = Options(**{name: getattr(args, name) for name in Options.__dataclass_fields__})
        with redirect_stdout(sys.stderr):
            result = compile_kernel(args.kernel, arch=args.arch, opt_level=args.opt_level, options=options)
        _write_dumps([(path, content) for path, content in zip(paths, (result.optimized, result.unoptimized))
                      if path is not None])
        _write_ptx(result.ptx)
    except (OSError, ValueError, NotImplementedError, RuntimeError) as exc:
        print(f'{parser.prog}: error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f'{parser.prog}: interrupted', file=sys.stderr)
        return 130
    return 0
