from compiler.options import Hints
from compiler.passes.contraction import tensorize
from compiler.passes.recurrence import stage_recurrences

def run_passes(kernel, launch, options):
    if options.contraction_precision not in ('f32', 'tf32x3', 'bf16x3'):
        raise ValueError(f'unsupported contraction precision {options.contraction_precision}')
    contractions = tensorize(kernel, launch, options.contraction_precision) if options.contraction_precision != 'f32' else ()
    recurrences = stage_recurrences(kernel, launch) if options.recurrence_staging else ()
    return Hints(contractions, recurrences)
