from __future__ import annotations
import hashlib
import multiprocessing
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from llvmlite import binding as llvm
from llvmlite import ir
from compiler.cache import CompilationCache
from compiler.codegen.lower import Lowering
from compiler.options import Options
from compiler.passes.regforward import forward_reduction_loads_register
from compiler.passes.gridfold import fold_grid_contractions
from compiler.passes.recurrence import reorder_recurrences
from compiler.passes.vectorize import vectorize_elementwise
from compiler.ptx import rewrite_ptx
from compiler.storage import half_storage_types
from compiler.validation import parse_kernel, validate_compile_arguments

@dataclass(frozen=True)
class Compilation:
    ptx: str
    unoptimized: str
    optimized: str

_CACHE = CompilationCache()
_COMPILE_LOCK = threading.RLock()
_STAMP: bytes | None = None
_LLVM_READY = False

def _stamp() -> Any:
    global _STAMP
    if _STAMP is None:
        digest = hashlib.sha256()
        import emmy, llvmlite, numpy
        digest.update(emmy.__version__.encode() if hasattr(emmy, '__version__') else b'emmy')
        digest.update(llvmlite.__version__.encode())
        digest.update(numpy.__version__.encode())
        digest.update((sys.implementation.cache_tag or 'python').encode())
        digest.update(repr(llvm.llvm_version_info).encode())
        for name in sorted((Path(__file__).parent).rglob('*.py')):
            digest.update(name.relative_to(Path(__file__).parent).as_posix().encode() + b'\0')
            digest.update(name.read_bytes())
            digest.update(b'\0')
        _STAMP = digest.digest()
    return _STAMP

def clear_cache() -> None:
    with _COMPILE_LOCK:
        _CACHE.clear()

def _compile_one(job: Any) -> Compilation:
    path, arch, opt_level, options = job
    return compile_kernel(path, arch=arch, opt_level=opt_level, options=options)

def compile_many(jobs: Any, workers: int | None = None) -> list[Compilation]:
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError('workers must be a positive integer or None')
    try:
        jobs = list(jobs)
    except TypeError as exc:
        raise ValueError('jobs must be an iterable of (path, arch, opt_level, options)') from exc
    for index, job in enumerate(jobs):
        if not isinstance(job, (tuple, list)) or len(job) != 4:
            raise ValueError(f'job {index}: expected (path, arch, opt_level, options)')
    if not jobs:
        return []
    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if len(jobs) == 1 or workers == 1:
        return [_compile_one(job) for job in jobs]
    with multiprocessing.get_context('spawn').Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(_compile_one, jobs)

def compile_kernel(path: Path, *, arch: Any = 'sm_89', opt_level: int = 2, options: Options | None = None) -> Compilation:
    with _COMPILE_LOCK:
        return _compile_kernel(path, arch=arch, opt_level=opt_level, options=options)

def _compile_kernel(path: Path, *, arch: Any, opt_level: int, options: Options | None) -> Compilation:
    options = validate_compile_arguments(arch, opt_level, options)
    path = Path(path)
    payload = path.read_bytes()
    meta_payload = path.with_name('meta.json').read_bytes()
    kernel, launch = parse_kernel(payload, meta_payload)
    half_storage_types(kernel, launch)
    digest = hashlib.sha256(_stamp())
    for field in (payload, meta_payload, arch.encode(), bytes((opt_level,)), repr(sorted(options.__dict__.items())).encode()):
        digest.update(len(field).to_bytes(8, 'big'))
        digest.update(field)
    key = digest.hexdigest()
    root = os.environ.get('DEUTERON_CACHE')
    hit = _CACHE.get(key, root)
    if hit is not None:
        return Compilation(*hit)
    result = _compile_uncached(kernel, launch, arch, opt_level, options)
    _CACHE.put(key, (result.ptx, result.unoptimized, result.optimized), root)
    return result

def _compile_uncached(kernel: Any, launch: Any, arch: str, opt_level: int, options: Options) -> Compilation:
    global _LLVM_READY
    if options.elementwise_vectorize:
        vectorized = vectorize_elementwise(kernel, launch)
        if vectorized is not None:
            kernel = vectorized
    if options.register_forwarding:
        forwarded = forward_reduction_loads_register(kernel, launch)
        if forwarded is not None:
            kernel = forwarded
    if options.recurrence_staging:
        kernel = reorder_recurrences(kernel, launch)
    kernel = fold_grid_contractions(kernel, launch)
    if not _LLVM_READY:
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        _LLVM_READY = True
    features = '+ptx90' if arch == 'sm_110' else '+ptx87' if int(arch[3:]) >= 100 else '+ptx80'
    target = llvm.Target.from_triple('nvptx64-nvidia-cuda')
    machine = target.create_target_machine(cpu=arch, features=features, opt=opt_level)
    module: Any = ir.Module(name=kernel.name)
    module.triple = 'nvptx64-nvidia-cuda'
    module.data_layout = str(machine.target_data)
    lowering = Lowering(kernel, launch, module, options)
    unoptimized = lowering.generate()
    with llvm.create_context() as context, llvm.parse_assembly(unoptimized, context=context) as lowered:
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
    return Compilation(ptx, unoptimized, optimized)
