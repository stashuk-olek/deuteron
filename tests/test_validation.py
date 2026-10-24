import copy
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from compiler import Options, compile_kernel
from compiler.validation import parse_kernel, validate_launch


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'vector_add'
SOURCE = EXAMPLE / 'vector_add.ir.json'


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(SOURCE.read_text())
        self.metadata = json.loads((EXAMPLE / 'meta.json').read_text())

    def compile_modified(self, graph=None, metadata=None, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'kernel.ir.json'
            source.write_text(json.dumps(self.graph if graph is None else graph))
            source.with_name('meta.json').write_text(json.dumps(self.metadata if metadata is None else metadata))
            return compile_kernel(source, **kwargs)

    def set_body(self, expression):
        kernel = next(node for node in self.graph['nodes'].values() if node.get('op') == 'KernelOp')
        kernel['op_fields']['body'] = [expression]

    def test_extra_duplicate_argument_order_is_rejected(self):
        self.metadata['kernels'][0]['arg_order'].append('lhs')
        with self.assertRaisesRegex(ValueError, 'args and arg_order disagree'):
            self.compile_modified()

    def test_malformed_launches_fail_before_llvm_compilation(self):
        cases = (
            ('grid', [0, 1, 1]), ('grid', [2**31, 1, 1]), ('grid', [True, 1, 1]),
            ('block', [1024, 2, 1]), ('block', [32., 1, 1]), ('block', [32, 1]),
            ('smem_bytes', -1), ('smem_bytes', True), ('zero_outputs', ['lhs']),
            ('zero_outputs', ['sum', 'sum']), ('zero_outputs', ['missing']),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                metadata = copy.deepcopy(self.metadata)
                metadata['kernels'][0][field] = value
                with patch('compiler.build._compile_uncached') as backend:
                    with self.assertRaises(ValueError):
                        self.compile_modified(metadata=metadata)
                    backend.assert_not_called()

    def test_buffer_types_shapes_and_flags_are_validated(self):
        for field, value in (('dtype', 'bf16'), ('shape', [-1]), ('shape', [True]), ('shape', '1030'), ('output', 1), ('intermediate', 'yes')):
            with self.subTest(field=field, value=value):
                metadata = copy.deepcopy(self.metadata)
                metadata['kernels'][0]['args'][0][field] = value
                with self.assertRaises(ValueError):
                    self.compile_modified(metadata=metadata)

    def test_readonly_global_write_is_rejected_for_f32(self):
        self.metadata['kernels'][0]['args'][-1]['output'] = False
        with self.assertRaisesRegex(NotImplementedError, 'output buffers'):
            self.compile_modified()

    def test_shape_rank_mismatch_is_rejected(self):
        self.metadata['kernels'][0]['args'][0]['shape'] = [2, 515]
        with self.assertRaisesRegex(ValueError, 'index rank'):
            self.compile_modified()

    def test_api_options_reject_invalid_values(self):
        for field, value in (('unroll', -1), ('unroll', 2**31), ('unroll', True), ('vector_width', 3), ('vector_width', 4.), ('contraction_precision', 'bf16'), ('approximate_math', 1), ('register_forwarding', None)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                invalid: dict[str, Any] = {field: value}
                Options(**invalid)
        for kwargs in ({'arch': None}, {'arch': 120}, {'opt_level': True}, {'opt_level': 1.0}, {'opt_level': -1}, {'options': {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compile_modified(**kwargs)

    def test_mutated_frozen_options_are_revalidated(self):
        options = Options()
        object.__setattr__(options, 'unroll', -1)
        with self.assertRaises(ValueError):
            compile_kernel(SOURCE, options=options)

    def test_api_retains_non_power_of_two_unroll_hints(self):
        compiled = compile_kernel(SOURCE, options=Options(unroll=3))
        self.assertIn('.entry vector_add', compiled.ptx)

    def test_constructor_escape_never_executes(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'must-not-exist'
            escape = "Literal.__init__.__globals__['__builtins__']['__import__']('pathlib').Path(" + repr(str(marker)) + ").write_text('executed')"
            self.set_body('Literal(value=' + escape + ", dtype='int')")
            with self.assertRaises(ValueError):
                self.compile_modified()
            self.assertFalse(marker.exists())

    def test_arbitrary_python_forms_are_rejected(self):
        expressions = (
            "__import__('os')", 'Literal.__init__', '(lambda: 0)()',
            '[Literal(i) for i in range(2)]', 'Literal(*[1])',
            "Literal(**{'value': 1})", '[Literal(1)][0]', 'UnknownStatement()',
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                self.set_body(expression)
                with self.assertRaises(ValueError):
                    self.compile_modified()

    def test_duplicate_json_keys_are_rejected(self):
        metadata = json.dumps(self.metadata).encode()
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            parse_kernel(b'{"nodes": {}, "nodes": {}}', metadata)

    def test_malformed_json_and_graph_have_value_errors(self):
        for payload in (b'[]', b'{', b'{"nodes": []}', b'{"nodes": {}}', b'\xff'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_kernel(payload, json.dumps(self.metadata).encode())
        with self.assertRaises(ValueError):
            parse_kernel(json.dumps(self.graph).encode(), b'{}')

    def test_compiler_does_not_call_the_emmy_eval_loader(self):
        with patch('emmy.compiler.graph.Graph.from_dict', side_effect=AssertionError('unsafe loader reached')):
            compiled = self.compile_modified(opt_level=0)
        self.assertIn('.entry vector_add', compiled.ptx)

    def test_runtime_schema_retains_tuple_and_empty_buffer_support(self):
        launch = copy.deepcopy(self.metadata['kernels'][0])
        launch['grid'] = tuple(launch['grid'])
        launch['block'] = tuple(launch['block'])
        for arg in launch['args']:
            arg['shape'] = (0,)
        self.assertEqual(set(validate_launch(launch)), {'lhs', 'rhs', 'sum'})


if __name__ == '__main__':
    unittest.main()
