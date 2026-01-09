"""Mutation Testing Library - Runtime mutation injection for Python tests."""

from .core import (
    Mutation,
    MutationError,
    MutationInjector,
    MutationResult,
    run_mutation_tests,
)

__version__ = "0.1.0"
__all__ = [
    "Mutation",
    "MutationError",
    "MutationInjector",
    "MutationResult",
    "run_mutation_tests",
]
