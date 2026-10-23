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

