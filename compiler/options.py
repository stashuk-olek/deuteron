from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Options:
    approximate_math: bool = True
    contraction_precision: str = 'bf16x3'
    recurrence_staging: bool = True
    reduction_sharing: bool = True
    vector_width: int = 4
    unroll: int = 0
    tree_specialization: bool = True
    coalesced_reductions: bool = True
    warp_willreturn: bool = True
    elementwise_vectorize: bool = True
    register_forwarding: bool = True
    tail_linearize: bool = True

    def __post_init__(self) -> None:
        for name in ('approximate_math', 'recurrence_staging', 'reduction_sharing', 'tree_specialization', 'coalesced_reductions', 'warp_willreturn', 'elementwise_vectorize', 'register_forwarding', 'tail_linearize'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if type(self.contraction_precision) is not str or self.contraction_precision not in ('f32', 'tf32x3', 'bf16x3'):
            raise ValueError(f'unsupported contraction precision {self.contraction_precision!r}')
        if type(self.vector_width) is not int or self.vector_width not in (1, 2, 4, 8, 16):
            raise ValueError('vector_width must be 1, 2, 4, 8, or 16')
        if type(self.unroll) is not int or not 0 <= self.unroll < 2**31:
            raise ValueError('unroll must be a nonnegative signed 32-bit integer')

@dataclass(frozen=True)
class Contraction:
    tile: Any
    loop: Any
    outer: object
    lhs_buffer: str
    lhs_base: tuple
    rhs_buffer: str
    rhs_base: tuple
    output_buffer: str
    output_base: tuple
    atomic: bool
    rows: int
    cols: int
    depth: int
    writes: tuple

@dataclass
class Recurrence:
    tile: Any
    loop: Any
    prefix: tuple
    suffix: tuple
    reductions: tuple
    invariant: tuple
    varying: tuple
    scale: str
    width: int
    extent: int
    components: tuple
    gather: tuple
    identity: float
    tag: int
    staged: tuple = ()

@dataclass(frozen=True)
class Hints:
    contractions: tuple = ()
    recurrences: tuple = ()

