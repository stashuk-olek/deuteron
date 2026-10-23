from __future__ import annotations
import hashlib
import json
import multiprocessing
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from emmy.compiler.graph import Graph
from llvmlite import binding as llvm
from llvmlite import ir
from compiler.codegen.lower import Lowering
from compiler.options import Options
from compiler.passes.vectorize import vectorize_elementwise
from compiler.ptx import rewrite_ptx

@dataclass(frozen=True)
class Compilation:
    ptx: str
    unoptimized: str
    optimized: str

_MEMORY: dict[str, Compilation] = {}
_STAMP: bytes | None = None

def _stamp() -> Any:
    global _STAMP
    if _STAMP is None:
        digest = hashlib.sha256()
        import emmy, llvmlite
        digest.update(emmy.__version__.encode() if hasattr(emmy, '__version__') else b'emmy')
        digest.update(llvmlite.__version__.encode())
        for name in sorted((Path(__file__).parent).rglob('*.py')):
            digest.update(name.read_bytes())
        _STAMP = digest.digest()
    return _STAMP

def _entry(key: str) -> Any:
    hit = _MEMORY.get(key)
    if hit is not None:
        return hit
    root = os.environ.get('DEUTERON_CACHE')
    if root:
        blob = Path(root) / (key + '.json')
        if blob.is_file():
            data = json.loads(blob.read_text())
            hit = Compilation(data['ptx'], data['unoptimized'], data['optimized'])
            _MEMORY[key] = hit
            return hit
    return None

def _store(key: str, result: Compilation) -> None:
    _MEMORY[key] = result
    root = os.environ.get('DEUTERON_CACHE')
    if root:
        blob = Path(root) / (key + '.json')
        blob.parent.mkdir(parents=True, exist_ok=True)
        tmp = blob.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'ptx': result.ptx, 'unoptimized': result.unoptimized, 'optimized': result.optimized}))
        os.replace(tmp, blob)

def _compile_one(job: Any) -> Compilation:
    path, arch, opt_level, options = job
    return compile_kernel(path, arch=arch, opt_level=opt_level, options=options)

def compile_many(jobs: Any, workers: int | None = None) -> list[Compilation]:
    jobs = list(jobs)
    if not jobs:
        return []
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers < 1:
        raise ValueError('workers must be >= 1')
    if len(jobs) == 1 or workers == 1:
        return [_compile_one(job) for job in jobs]
    with multiprocessing.get_context('spawn').Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(_compile_one, jobs)

def compile_kernel(path: Path, *, arch: Any = 'sm_89', opt_level: int = 2, options: Options | None = None) -> Compilation:
    if not re.fullmatch('sm_(?:8[069]|9[0]|1[012][0])', arch):
        raise ValueError(f'unsupported target {arch}; use sm_80, sm_86, sm_89, sm_90, sm_100, sm_110, or sm_120')
    if opt_level not in (0, 1, 2, 3):
        raise ValueError('optimization level must be 0, 1, 2, or 3')
    if options is None:
        options = Options()
    path = Path(path)
    payload = path.read_bytes()
    meta_payload = path.with_name('meta.json').read_bytes()
    digest = hashlib.sha256(_stamp())
    digest.update(payload)
    digest.update(meta_payload)
    digest.update(arch.encode())
    digest.update(bytes((opt_level,)))
    digest.update(repr(sorted(options.__dict__.items())).encode())
    key = digest.hexdigest()
    hit = _entry(key)
    if hit is not None:
        return hit
    graph = Graph.from_dict(json.loads(payload))
    kernels: list[Any] = [n.op for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp']
    if len(kernels) != 1:
        raise ValueError('expected exactly one KernelOp in the input graph')
    kernel = kernels[0]
    metadata = json.loads(meta_payload)
    launches = [k for k in metadata['kernels'] if k['entry'] == kernel.name]
    if len(launches) != 1:
        raise ValueError(f'meta.json must contain exactly one launch for {kernel.name}')
    if options.elementwise_vectorize:
        vectorized = vectorize_elementwise(kernel, launches[0])
        if vectorized is not None:
            kernel = vectorized
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    features = '+ptx87' if int(arch[3:]) >= 100 else '+ptx80'
    target = llvm.Target.from_triple('nvptx64-nvidia-cuda')
    machine = target.create_target_machine(cpu=arch, features=features, opt=opt_level)
    module: Any = ir.Module(name=kernel.name)
    module.triple = 'nvptx64-nvidia-cuda'
    module.data_layout = str(machine.target_data)
    lowering = Lowering(kernel, launches[0], module, options)
    unoptimized = lowering.generate()
    with llvm.parse_assembly(unoptimized) as lowered:
        lowered.verify()
        tuning = llvm.PipelineTuningOptions(speed_level=opt_level)
        tuning.loop_vectorization = options.vector_width > 1
        tuning.slp_vectorization = opt_level >= 2
        with llvm.create_pass_builder(machine, tuning) as passes:
            with llvm.create_new_module_pass_manager() as promotion:
                promotion.add_sroa_pass()
                promotion.run(lowered, passes)
            with passes.getModulePassManager() as optimizer:
                optimizer.run(lowered, passes)
            if opt_level:
                with llvm.create_new_module_pass_manager() as cleanup:
                    if lowering.striped:
                        cleanup.add_loop_strength_reduce_pass()
                    cleanup.add_new_gvn_pass()
                    cleanup.add_instruction_combine_pass()
                    cleanup.add_simplify_cfg_pass()
                    cleanup.run(lowered, passes)
        lowered.verify()
        optimized = str(lowered)
        ptx = rewrite_ptx(machine.emit_assembly(lowered), options.approximate_math)
    result = Compilation(ptx, unoptimized, optimized)
    _store(key, result)
    return result
