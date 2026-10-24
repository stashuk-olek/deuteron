import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import secrets
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import RemoteRunner, compare_outputs


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


class CachedRunner(RemoteRunner):
    def __init__(self, server):
        super().__init__(server)
        self.cache = {}

    def request(self, endpoint, body):
        if endpoint != '/compile':
            return super().request(endpoint, body)
        key = sha(body['ptx'].encode())
        if key not in self.cache:
            target = re.search(r'^\.target\s+(sm_[0-9]+a?)', body['ptx'], re.M).group(1)
            self.cache[key] = super().request(endpoint, {**body, 'arch': target if target.endswith('a') else 'sm_120'})
        return self.cache[key]


def measure(runner, item, problems):
    directory = problems / item['problem']
    meta = json.loads((directory / 'meta.json').read_text())
    for filename, field in (('meta.json', 'meta_sha256'), ('expected.npz', 'expected_sha256')):
        if sha((directory / filename).read_bytes()) != item[field]:
            raise ValueError(f'Changed {directory / filename}')
    ptx = Path(item['ptx']).read_text()
    if sha(ptx.encode()) != item['ptx_sha256']:
        raise ValueError(f'Changed PTX for {item["problem"]}/{item["compiler"]}')
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    response = runner.run_kernel(item['launch'], ptx, warmup=20, iters=200)
    elapsed = time.perf_counter() - before
    actual = response.pop('outputs')
    compare_outputs(directory, meta, actual)
    errors = {}
    with np.load(directory / 'expected.npz') as reference:
        for name in reference.files:
            delta = np.abs(actual[name].astype(np.float64) - reference[name].astype(np.float64))
            allowance = meta['atol'] + meta['rtol'] * np.abs(reference[name].astype(np.float64))
            ratio = np.divide(delta, allowance, out=np.where(delta == 0, 0., np.inf), where=allowance != 0)
            errors[name] = {'max_absolute_error': float(np.max(delta)), 'max_normalized_tolerance_ratio': float(np.max(ratio)), 'output_sha256': sha(actual[name].tobytes())}
    return {**response, 'started_utc': started, 'request_seconds': elapsed, 'correct': True, 'errors': errors}


def screen(args, items, runner):
    records = []
    for item in items:
        calls = [measure(runner, item, args.problems_dir) for _ in range(2 if item['problem'].startswith(('a', 'm')) else 1)]
        records.append({'variant': item, 'calls': calls, 'warm_median_us': statistics.median(c['median_us'] for c in calls)})
        save(args.out, {'purpose': 'Configuration screen; excluded from the final comparison', 'records': records})
        print(item['problem'], item['config'], records[-1]['warm_median_us'], flush=True)
    selected = []
    for name in sorted({item['problem'] for item in items}):
        selected.append(min((r for r in records if r['variant']['problem'] == name), key=lambda r: (r['warm_median_us'], r['variant']['config']))['variant'])
    save(args.out.with_name(args.out.stem + '-selected.json'), selected)


def compare(args, items, runner):
    names = sorted({item['problem'] for item in items})
    compilers = sorted({item['compiler'] for item in items})
    if len(compilers) != 3:
        raise ValueError('Provide exactly three compiler manifests')
    by_key = {(item['problem'], item['compiler']): item for item in items}
    if len(by_key) != len(items) or set(by_key) != set(itertools.product(names, compilers)):
        raise ValueError('Expected one selected variant per problem and compiler')
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    rng = random.Random(seed)
    permutations = list(itertools.permutations(compilers))
    orders = {name: rng.sample(permutations, len(permutations)) for name in names}
    rounds = [[{'problem': name, 'order': orders[name][round_number]} for name in rng.sample(names, len(names))] for round_number in range(6)]
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'warmup': 20, 'iters': 200, 'rounds': rounds, 'samples_per_variant': 6, 'policy': 'All six compiler permutations per problem; all samples retained; configuration screen excluded; no early stopping or resampling.'}
    schedule = args.out.with_name(args.out.stem + '-schedule.json')
    save(schedule, protocol)
    results = {'status': 'running', 'server': args.server, 'protocol': protocol, 'protocol_sha256': sha(schedule.read_bytes()), 'variants': items, 'calls': [], 'harness_sha256': sha(Path(__file__).read_bytes())}
    save(args.out, results)
    try:
        for round_number, round_items in enumerate(rounds):
            for group in round_items:
                for position, compiler in enumerate(group['order']):
                    item = by_key[group['problem'], compiler]
                    call = {'round': round_number, 'position': position, 'problem': item['problem'], 'compiler': compiler, **measure(runner, item, args.problems_dir)}
                    results['calls'].append(call)
                    save(args.out, results)
                print('round', round_number + 1, group['problem'], {c['compiler']: c['median_us'] for c in results['calls'][-3:]}, flush=True)
        summary = {}
        for name in names:
            summary[name] = {}
            for compiler in compilers:
                calls = [c for c in results['calls'] if c['problem'] == name and c['compiler'] == compiler]
                values = [c['median_us'] for c in calls]
                if len(values) != 6:
                    raise ValueError('Incomplete comparison')
                summary[name][compiler] = {'median_us': statistics.median(values), 'min_us': min(values), 'max_us': max(values), 'series_us': values}
        results.update(status='complete', summary=summary, geometric_mean_latency_us={compiler: math.exp(statistics.fmean(math.log(summary[name][compiler]['median_us']) for name in names)) for compiler in compilers})
    except BaseException as error:
        results.update(status='failed', error=repr(error))
        raise
    finally:
        results['ended_utc'] = datetime.now(timezone.utc).isoformat()
        results['compile_records'] = runner.cache
        save(args.out, results)


def main():
    parser = argparse.ArgumentParser(description='Correctness-checked RTX 5090 comparison of three prepared compiler baselines')
    parser.add_argument('mode', choices=('screen', 'run'))
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--problems-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server', default='http://runner.riftstack.ai:48701')
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f'Use a new output file: {args.out}')
    items = []
    for manifest in args.manifest:
        for item in json.loads(manifest.read_text()):
            item['ptx'] = str((manifest.parent / item['ptx']).resolve())
            items.append(item)
    lock = args.lock or args.out.parent / 'runner.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        runner = CachedRunner(args.server)
        (screen if args.mode == 'screen' else compare)(args, items, runner)


if __name__ == '__main__':
    main()
