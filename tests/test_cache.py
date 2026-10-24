from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import traceback
import unittest
from unittest import mock

from compiler import compile_kernel, compile_many
from compiler import build
from compiler.build import clear_cache
from compiler.cache import CompilationCache, _record


ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / 'examples/vector_add/vector_add.ir.json'
MATRIX = ROOT / 'examples/tensor_matmul/matmul_tiled_128.ir.json'
KEY = hashlib.sha256(b'cache-test').hexdigest()
VALUE = (
    '.version 8.0\n.target sm_89\n.address_size 64\n.visible .entry kernel() { ret; }\n',
    'target triple = "nvptx64-nvidia-cuda"\ndefine ptx_kernel void @kernel() { ret void }\n',
    'target triple = "nvptx64-nvidia-cuda"\ndefine ptx_kernel void @kernel() { ret void }\n',
)


def _cold_compile(path, directory, barrier, results):
    try:
        os.environ['DEUTERON_CACHE'] = directory
        clear_cache()
        barrier.wait(timeout=30)
        result = compile_kernel(Path(path), opt_level=3)
        results.put({'pid': os.getpid(), 'sha256': hashlib.sha256(result.ptx.encode()).hexdigest()})
    except BaseException:
        results.put({'pid': os.getpid(), 'error': traceback.format_exc()})
        raise


class CacheRecordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='deuteron-cache-test-')
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.path = Path(self.directory) / (KEY + '.json')

    def test_disk_roundtrip_and_memory_clear(self):
        cache = CompilationCache()
        cache.put(KEY, VALUE, self.directory)
        self.assertEqual(cache.get(KEY), VALUE)
        cache.clear()
        self.assertIsNone(cache.get(KEY))
        self.assertEqual(cache.get(KEY, self.directory), VALUE)
        self.assertEqual(set(Path(self.directory).iterdir()), {self.path})

    def test_corrupt_and_stale_records_are_misses(self):
        record = _record(KEY, VALUE)
        mutations = [
            b'{', b'\xff', b'null', b'[]', b'[' * 1500 + b']' * 1500,
            json.dumps({field: record[field] for field in ('ptx', 'unoptimized', 'optimized')}).encode(),
        ]
        for field, value in (
            ('version', 0), ('version', True), ('version', '1'), ('key', '0' * 64),
            ('ptx', None), ('optimized', []), ('unoptimized', ''), ('sha256', 'f' * 64),
            ('sha256', []), ('unexpected', 1), ('ptx', VALUE[0] + '// corruption'),
        ):
            mutations.append(json.dumps({**record, field: value}).encode())
        for index, payload in enumerate(mutations):
            with self.subTest(index=index):
                self.path.write_bytes(payload)
                self.assertIsNone(CompilationCache().get(KEY, self.directory))

    def test_rechecks_shapes_even_with_a_matching_checksum(self):
        for value in (('not PTX', *VALUE[1:]), (VALUE[0], 'not LLVM', VALUE[2])):
            with self.subTest(value=value):
                self.path.write_text(json.dumps(_record(KEY, value)))
                self.assertIsNone(CompilationCache().get(KEY, self.directory))

    def test_record_size_is_bounded(self):
        self.path.write_bytes(b' ' * 257)
        with mock.patch('compiler.cache._MAX_RECORD_BYTES', 256):
            self.assertIsNone(CompilationCache().get(KEY, self.directory))

    def test_io_failures_are_best_effort_and_leave_no_temporary_files(self):
        cache = CompilationCache()
        with mock.patch('compiler.cache.os.replace', side_effect=PermissionError('readonly')):
            cache.put(KEY, VALUE, self.directory)
        self.assertEqual(cache.get(KEY), VALUE)
        self.assertEqual(list(Path(self.directory).iterdir()), [])
        cache.clear()
        with mock.patch('compiler.cache.Path.open', side_effect=PermissionError('unreadable')):
            self.assertIsNone(cache.get(KEY, self.directory))
        cache.put(KEY, VALUE, '\0')
        cache.clear()
        self.assertIsNone(cache.get(KEY, '\0'))

    def test_lru_entry_and_byte_limits(self):
        keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(3)]
        cache = CompilationCache(max_entries=2)
        cache.put(keys[0], VALUE)
        cache.put(keys[1], VALUE)
        self.assertEqual(cache.get(keys[0]), VALUE)
        cache.put(keys[2], VALUE)
        self.assertIsNone(cache.get(keys[1]))
        self.assertEqual(cache.get(keys[0]), VALUE)
        size = sum(sys.getsizeof(text) for text in VALUE)
        cache = CompilationCache(max_bytes=size)
        cache.put(keys[0], VALUE)
        cache.put(keys[1], VALUE)
        self.assertIsNone(cache.get(keys[0]))
        self.assertEqual(cache.get(keys[1]), VALUE)
        large = (VALUE[0] + 'extra', *VALUE[1:])
        cache.put(keys[2], large)
        self.assertIsNone(cache.get(keys[2]))
        self.assertEqual(cache.get(keys[1]), VALUE)


class CompilerCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='deuteron-compiler-cache-')
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        environment = mock.patch.dict(os.environ, {'DEUTERON_CACHE': self.directory})
        environment.start()
        self.addCleanup(environment.stop)
        clear_cache()
        self.addCleanup(clear_cache)

    def test_persistent_hit_skips_codegen(self):
        expected = compile_kernel(VECTOR, opt_level=3)
        clear_cache()
        with mock.patch.object(build, '_compile_uncached', side_effect=AssertionError('cache miss')):
            self.assertEqual(compile_kernel(VECTOR, opt_level=3), expected)

    def test_corruption_is_recompiled_and_repaired(self):
        expected = compile_kernel(VECTOR, opt_level=3)
        path, = Path(self.directory).glob('*.json')
        path.write_text('{truncated')
        clear_cache()
        with mock.patch.object(build, '_compile_uncached', wraps=build._compile_uncached) as codegen:
            self.assertEqual(compile_kernel(VECTOR, opt_level=3), expected)
            self.assertEqual(codegen.call_count, 1)
        clear_cache()
        with mock.patch.object(build, '_compile_uncached', side_effect=AssertionError('repair failed')):
            self.assertEqual(compile_kernel(VECTOR, opt_level=3), expected)

    def test_unwritable_cache_does_not_fail_compilation(self):
        expected = compile_kernel(VECTOR, opt_level=3)
        clear_cache()
        blocker = Path(self.directory) / 'file'
        blocker.write_text('not a directory')
        os.environ['DEUTERON_CACHE'] = str(blocker)
        self.assertEqual(compile_kernel(VECTOR, opt_level=3), expected)

    def test_cache_key_distinguishes_targets_and_levels(self):
        jobs = [(VECTOR, 'sm_89', 2, None), (VECTOR, 'sm_89', 3, None), (VECTOR, 'sm_120', 3, None)]
        for path, arch, level, options in jobs:
            compile_kernel(path, arch=arch, opt_level=level, options=options)
        self.assertEqual(len(list(Path(self.directory).glob('*.json'))), len(jobs))

    def test_concurrent_threads_compile_a_cold_key_once(self):
        barrier = threading.Barrier(8)

        def run(_):
            barrier.wait(timeout=30)
            return compile_kernel(MATRIX, opt_level=3)

        with mock.patch.object(build, '_compile_uncached', wraps=build._compile_uncached) as codegen:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(run, range(8)))
            self.assertEqual(codegen.call_count, 1)
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(list(Path(self.directory).glob('*.json'))), 1)

    def test_distinct_threaded_compiles_match_serial_results(self):
        jobs = [('sm_89', 2), ('sm_89', 3), ('sm_120', 2), ('sm_120', 3)]
        expected = [compile_kernel(VECTOR, arch=arch, opt_level=level) for arch, level in jobs]
        clear_cache()
        os.environ.pop('DEUTERON_CACHE')
        barrier = threading.Barrier(len(jobs))

        def run(job):
            barrier.wait(timeout=30)
            return compile_kernel(VECTOR, arch=job[0], opt_level=job[1])

        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            self.assertEqual(list(pool.map(run, jobs)), expected)

    def test_simultaneous_cold_processes_publish_one_complete_record(self):
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(5)
        results = context.Queue()
        processes = [context.Process(target=_cold_compile, args=(str(MATRIX), self.directory, barrier, results)) for _ in range(4)]
        try:
            for process in processes:
                process.start()
            barrier.wait(timeout=30)
            returned = [results.get(timeout=60) for _ in processes]
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            self.assertFalse([value for value in returned if 'error' in value], returned)
            self.assertEqual(len({value['pid'] for value in returned}), 4)
            self.assertEqual(len({value['sha256'] for value in returned}), 1)
            path, = Path(self.directory).iterdir()
            self.assertEqual(path.suffix, '.json')
            value = CompilationCache().get(path.stem, self.directory)
            self.assertIsNotNone(value)
            self.assertEqual(hashlib.sha256(value[0].encode()).hexdigest(), returned[0]['sha256'])
        except (queue.Empty, threading.BrokenBarrierError) as exc:
            self.fail(str(exc))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(timeout=10)
            results.close()
            results.join_thread()

    def test_compile_many_spawn_workers_share_the_persistent_cache(self):
        jobs = [(VECTOR, 'sm_89', 3, None)] * 6
        values = compile_many(jobs, workers=3)
        self.assertTrue(all(value == values[0] for value in values))
        self.assertEqual(len(list(Path(self.directory).glob('*.json'))), 1)
        clear_cache()
        with mock.patch.object(build, '_compile_uncached', side_effect=AssertionError('missing process cache')):
            self.assertEqual(compile_kernel(VECTOR, opt_level=3), values[0])

    def test_invalid_workers_reject_even_without_jobs(self):
        for workers in (False, True, 1.5, '2', 0, -1):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'workers'):
                compile_many([], workers=workers)
        self.assertEqual(compile_many([], workers=None), [])

    def test_invalid_job_shapes_report_value_errors_before_compiling(self):
        valid = (VECTOR, 'sm_89', 3, None)
        for jobs in (None, ['abcd'], [()], [valid, (VECTOR, 'sm_89', 3)], [dict(enumerate(valid))]):
            with self.subTest(jobs=jobs), mock.patch.object(build, '_compile_uncached', side_effect=AssertionError('compiled malformed jobs')):
                with self.assertRaisesRegex(ValueError, 'job'):
                    compile_many(jobs, workers=1)


if __name__ == '__main__':
    unittest.main()
