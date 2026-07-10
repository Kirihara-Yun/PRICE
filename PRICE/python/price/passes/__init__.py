"""Graph transformation passes for PRICE.

This package collects the building blocks used by the optimizer to explore
recomputation / fission schedules:

* ``GraphTransformer`` and its mutator subclasses generate candidate graphs.
* ``TransformRule`` and the rule sets under ``rewrite_rules`` encode
  algebraic equivalences that the mutators may apply.
"""
from .mutator import (
    GraphTransformer,
    ChainedMutator,
    ZippedMutator,
    TruncatedMutator,
    RandomMutator,
    TransformRuleMutator,
    SwapRematMutator,
    FissionFactorMutator,
    FissionTreeMutator,
)
from .rewrite_rules import TransformRule
from . import rewrite_rules as RULES
from .misc import (
    apply_fission,
    remove_fission,
    apply_swap,
    apply_remat,
)
