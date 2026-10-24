from contextlib import redirect_stderr, redirect_stdout
from dataclasses import fields
import errno
from io import StringIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from compiler import Compilation, Options, compile_kernel
from compiler import cli
from compiler._version import __version__


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'compile.py'
EXAMPLE = ROOT / 'examples' / 'vector_add'
STUB = Compilation('// PTX only\n', '; unoptimized\n', '; optimized\n')


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.kernel = self.directory / 'vector_add.ir.json'
        self.meta = self.directory / 'meta.json'
        shutil.copyfile(EXAMPLE / self.kernel.name, self.kernel)
        shutil.copyfile(EXAMPLE / 'meta.json', self.meta)
        self.environment = os.environ.copy()
        for name in ('PYTHONPATH', 'DEUTERON_CACHE', 'DEUTERON_TEST_DEVICE'):
            self.environment.pop(name, None)

    def process(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              cwd=self.directory, env=self.environment,
                              capture_output=True, text=True, timeout=60)

    def invoke(self, args):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(list(map(str, args)))
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_failure(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertTrue(result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_original_interface_defaults_and_dump_contents(self):
        optimized = self.directory / 'optimized.ll'
        unoptimized = self.directory / 'unoptimized.ll'
        result = self.process(self.kernel, '--dump-llvm', optimized, '--dump-unoptimized', unoptimized)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        expected = compile_kernel(self.kernel, arch='sm_89', opt_level=3, options=Options())
        self.assertEqual(result.stdout, expected.ptx)
        self.assertEqual(optimized.read_text(), expected.optimized)
        self.assertEqual(unoptimized.read_text(), expected.unoptimized)
        self.assertIn('.target sm_89', result.stdout)

    def test_all_existing_options_are_forwarded(self):
        args = [self.kernel, '--arch', 'sm_120', '--opt-level', '1', '--unroll', '8',
                '--vector-width', '16', '--contraction-precision', 'f32']
        values = {'unroll': 8, 'vector_width': 16, 'contraction_precision': 'f32'}
        for field in fields(Options):
            if isinstance(field.default, bool):
                args.append('--no-' + field.name.replace('_', '-'))
                values[field.name] = False
        with patch.object(cli, 'compile_kernel', return_value=STUB) as compile_mock:
            code, stdout, stderr = self.invoke(args)
        self.assertEqual((code, stdout, stderr), (0, STUB.ptx, ''))
        compile_mock.assert_called_once_with(self.kernel, arch='sm_120', opt_level=1, options=Options(**values))

    def test_help_and_version_need_no_kernel(self):
        help_result = self.process('--help')
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for field in fields(Options):
            self.assertIn('--' + field.name.replace('_', '-'), help_result.stdout)
        version_result = self.process('--version')
        self.assertEqual(version_result.returncode, 0, version_result.stderr)
        self.assertEqual(version_result.stdout, f'deuteron {__version__}\n')
        self.assertEqual(version_result.stderr, '')

    def test_invalid_command_line_is_reported_without_ptx(self):
        for args in ((), (self.kernel, '--opt-level', '4'), (self.kernel, '--arch', 'sm_999')):
            with self.subTest(args=args):
                self.assert_failure(self.process(*args))

    def test_missing_input_and_metadata_are_clear_failures(self):
        missing = self.directory / 'absent.ir.json'
        result = self.process(missing)
        self.assert_failure(result)
        self.assertIn('absent.ir.json', result.stderr)
        self.meta.unlink()
        result = self.process(self.kernel)
        self.assert_failure(result)
        self.assertIn('meta.json', result.stderr)

    def test_malformed_json_and_invalid_encoding_are_clear_failures(self):
        for content in (b'{"nodes":', b'\xff'):
            with self.subTest(content=content):
                self.kernel.write_bytes(content)
                self.assert_failure(self.process(self.kernel))
        shutil.copyfile(EXAMPLE / self.kernel.name, self.kernel)
        self.meta.write_text('{')
        self.assert_failure(self.process(self.kernel))

    def test_compiler_diagnostics_go_to_stderr(self):
        def compile_with_diagnostic(*args, **kwargs):
            print('compiler diagnostic')
            return STUB

        with patch.object(cli, 'compile_kernel', side_effect=compile_with_diagnostic):
            code, stdout, stderr = self.invoke([self.kernel])
        self.assertEqual(code, 0)
        self.assertEqual(stdout, STUB.ptx)
        self.assertEqual(stderr, 'compiler diagnostic\n')

    def test_input_and_metadata_dump_collisions_are_rejected(self):
        original = {path: path.read_bytes() for path in (self.kernel, self.meta)}
        for protected in original:
            for flag in ('--dump-llvm', '--dump-unoptimized'):
                with self.subTest(protected=protected, flag=flag):
                    with patch.object(cli, 'compile_kernel', return_value=STUB) as compile_mock:
                        code, stdout, stderr = self.invoke([self.kernel, flag, protected])
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout, '')
                    self.assertIn('overwrite an input', stderr)
                    compile_mock.assert_not_called()
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_symlink_and_hardlink_input_collisions_are_rejected(self):
        symlink = self.directory / 'symbolic.ll'
        hardlink = self.directory / 'hard.ll'
        symlink.symlink_to(self.kernel)
        os.link(self.meta, hardlink)
        original = {path: path.read_bytes() for path in (self.kernel, self.meta)}
        for alias in (symlink, hardlink):
            with self.subTest(alias=alias), patch.object(cli, 'compile_kernel', return_value=STUB) as compile_mock:
                code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', alias])
                self.assertEqual(code, 1)
                self.assertEqual(stdout, '')
                self.assertIn('overwrite an input', stderr)
                compile_mock.assert_not_called()
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_dump_files_cannot_alias_each_other(self):
        first = self.directory / 'first.ll'
        first.write_text('old contents')
        hardlink = self.directory / 'alias.ll'
        os.link(first, hardlink)
        for second in (first, first.parent / '.' / first.name, hardlink):
            with self.subTest(second=second), patch.object(cli, 'compile_kernel', return_value=STUB) as compile_mock:
                code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', first, '--dump-unoptimized', second])
                self.assertEqual(code, 1)
                self.assertEqual(stdout, '')
                self.assertIn('different files', stderr)
                compile_mock.assert_not_called()
                self.assertEqual(first.read_text(), 'old contents')

    def test_invalid_dump_destinations_do_not_touch_existing_output(self):
        existing = self.directory / 'existing.ll'
        existing.write_text('keep me')
        for invalid in (self.directory, self.directory / 'absent' / 'dump.ll'):
            with self.subTest(invalid=invalid), patch.object(cli, 'compile_kernel', return_value=STUB) as compile_mock:
                code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', existing, '--dump-unoptimized', invalid])
                self.assertEqual(code, 1)
                self.assertEqual(stdout, '')
                self.assertTrue(stderr)
                self.assertEqual(existing.read_text(), 'keep me')
                compile_mock.assert_not_called()

    def test_compile_failure_does_not_modify_dump_files(self):
        destination = self.directory / 'dump.ll'
        destination.write_text('old contents')
        with patch.object(cli, 'compile_kernel', side_effect=ValueError('invalid scheduled IR')):
            code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', destination])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, '')
        self.assertIn('invalid scheduled IR', stderr)
        self.assertEqual(destination.read_text(), 'old contents')

    def test_all_dumps_are_staged_before_existing_outputs_are_replaced(self):
        first, second = self.directory / 'first.ll', self.directory / 'second.ll'
        first.write_text('old first')
        second.write_text('old second')
        before = set(self.directory.iterdir())
        factory = tempfile.NamedTemporaryFile
        calls = 0

        def fail_second_stage(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError(errno.EACCES, 'write denied', str(second))
            return factory(*args, **kwargs)

        with patch.object(cli, 'compile_kernel', return_value=STUB), patch.object(cli.tempfile, 'NamedTemporaryFile', side_effect=fail_second_stage):
            code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', first, '--dump-unoptimized', second])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, '')
        self.assertIn('write denied', stderr)
        self.assertEqual(first.read_text(), 'old first')
        self.assertEqual(second.read_text(), 'old second')
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_failed_atomic_replace_preserves_existing_file_and_cleans_temporary(self):
        destination = self.directory / 'dump.ll'
        destination.write_text('old contents')
        before = set(self.directory.iterdir())
        with patch.object(cli, 'compile_kernel', return_value=STUB), patch.object(Path, 'replace', side_effect=OSError(errno.ENOSPC, 'No space left')):
            code, stdout, stderr = self.invoke([self.kernel, '--dump-llvm', destination])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, '')
        self.assertIn('No space left', stderr)
        self.assertEqual(destination.read_text(), 'old contents')
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_broken_stdout_pipe_is_a_clean_failure(self):
        with subprocess.Popen([sys.executable, str(SCRIPT), str(self.kernel)],
                              cwd=self.directory, env=self.environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            assert process.stdout is not None and process.stderr is not None
            process.stdout.close()
            stderr = process.stderr.read()
            code = process.wait(timeout=60)
        self.assertEqual(code, 1, stderr)
        self.assertIn('Broken pipe', stderr)
        self.assertNotIn('Traceback', stderr)
        self.assertNotIn('Exception ignored', stderr)

    @unittest.skipUnless(Path('/dev/full').exists(), 'requires the POSIX full device')
    def test_stdout_flush_failure_is_reported_once(self):
        with open('/dev/full', 'wb') as stream:
            result = subprocess.run([sys.executable, str(SCRIPT), str(self.kernel)],
                                    cwd=self.directory, env=self.environment,
                                    stdout=stream, stderr=subprocess.PIPE, text=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('No space left', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertNotIn('Exception ignored', result.stderr)

    def test_interrupt_is_a_clean_nonzero_exit(self):
        with patch.object(cli, 'compile_kernel', side_effect=KeyboardInterrupt):
            code, stdout, stderr = self.invoke([self.kernel])
        self.assertEqual(code, 130)
        self.assertEqual(stdout, '')
        self.assertIn('interrupted', stderr)

    def test_packaging_and_requirements_have_the_same_pinned_dependencies(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        requirements = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines() if line.strip()]
        self.assertEqual(project['project']['dependencies'], requirements)
        self.assertEqual(project['project']['requires-python'], '>=3.12')
        self.assertEqual(project['project']['scripts']['deuteron'], 'compile:main')
        self.assertEqual(project['tool']['setuptools']['dynamic']['version'], {'attr': 'compiler._version.__version__'})


if __name__ == '__main__':
    unittest.main()
