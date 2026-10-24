import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
IR = Path('examples/tensor_matmul/matmul_tiled_128.ir.json')
ENV_UNSET = ('PYTHONPATH', 'DEUTERON_CACHE', 'DEUTERON_TEST_DEVICE', 'PTXAS')
ENV_SET = {'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compiler_hashes():
    paths = [ROOT / 'compile.py', *sorted((ROOT / 'compiler').rglob('*.py'))]
    return {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths}


def git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def first_instruction(ptx, pattern):
    for number, line in enumerate(ptx.splitlines(), 1):
        if re.match(pattern, line):
            return {'line': number, 'text': line.strip()}
    raise RuntimeError(f'expected instruction was not emitted: {pattern}')


def main():
    parser = argparse.ArgumentParser(description='Compile a tensor/copy example on the CPU and show two actual PTX instructions.')
    parser.add_argument('--output', required=True, type=Path, help='a new output directory; existing directories are refused')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'choose a fresh output directory: {output}')
    destination = output / 'kernel.ptx'
    argv = [sys.executable, 'compile.py', str(IR), '--arch', 'sm_120', '--opt-level', '3', '--contraction-precision', 'tf32x3']
    environment = dict(os.environ)
    for name in ENV_UNSET:
        environment.pop(name, None)
    environment.update(ENV_SET)
    display = ['env', *[part for name in ENV_UNSET for part in ('-u', name)], *[f'{name}={value}' for name, value in ENV_SET.items()], *argv]
    shown_destination = destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination
    command = shlex.join(display) + ' > ' + shlex.quote(str(shown_destination))
    print('$ ' + command, flush=True)
    before = compiler_hashes()
    sources = {str(path): digest((ROOT / path).read_bytes()) for path in (IR, IR.with_name('meta.json'))}
    revision = git('rev-parse', 'HEAD')
    revision = revision.decode().strip() if revision is not None else None
    release_hashes = {name: git('show', f'{revision}:{name}') for name in before} if revision else {}
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True)
    destination.write_bytes(result.stdout)
    (output / 'compiler.stderr.txt').write_bytes(result.stderr)
    after = compiler_hashes()
    evidence = {
        'schema_version': 1,
        'purpose': 'CPU-only compilation and inspection of actual emitted PTX',
        'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'checkout_commit': revision,
        'working_directory': str(ROOT),
        'command_argv': argv,
        'shell_command': command,
        'environment_unset': list(ENV_UNSET),
        'environment_set': ENV_SET,
        'python': sys.version,
        'dependencies': {name: importlib.metadata.version(name) for name in ('emmy-ml', 'llvmlite', 'numpy')},
        'compiler_files': before,
        'compiler_tree_sha256': digest(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()),
        'compiler_sources_unchanged': before == after,
        'compiler_matches_checkout_commit': all(data is not None and digest(data) == before[name] for name, data in release_hashes.items()) if revision else None,
        'input_files': sources,
        'inputs_unchanged': all(digest((ROOT / name).read_bytes()) == sha for name, sha in sources.items()),
        'exit_code': result.returncode,
        'ptx': 'kernel.ptx',
        'ptx_sha256': digest(result.stdout),
        'ptx_bytes': len(result.stdout),
        'stderr_sha256': digest(result.stderr),
        'assembly_performed': False,
        'gpu_execution_performed': False,
        'gpu_correctness_checked': False,
        'benchmark_performed': False,
    }
    transcript = '$ ' + command + '\n'
    if result.returncode == 0:
        ptx = result.stdout.decode()
        evidence['instructions'] = {
            'mma': first_instruction(ptx, r'^\s*(?:@!?%[\w.$]+\s+)?mma\.sync\.'),
            'cp_async': first_instruction(ptx, r'^\s*(?:@!?%[\w.$]+\s+)?cp\.async\.(?:ca|cg)\.shared\.global\b'),
        }
        for instruction in evidence['instructions'].values():
            print(instruction['text'], flush=True)
            transcript += instruction['text'] + '\n'
    else:
        sys.stderr.buffer.write(result.stderr)
    (output / 'transcript.txt').write_text(transcript)
    (output / 'command.txt').write_text(command + '\n')
    (output / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if not evidence['compiler_sources_unchanged'] or not evidence['inputs_unchanged']:
        raise RuntimeError('source or input files changed during compilation')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
