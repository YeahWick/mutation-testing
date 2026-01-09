"""Mutation Testing Library - Runtime mutation injection for Python tests."""

from .core import (
    Mutation,
    MutationError,
    MutationInjector,
    MutationResult,
    run_mutation_tests,
)
from .config import MutationConfig
from .runner import MutationRunner, MutationReport

__version__ = "0.1.0"
__all__ = [
    # Core
    "Mutation",
    "MutationError",
    "MutationInjector",
    "MutationResult",
    "run_mutation_tests",
    # High-level API
    "MutationConfig",
    "MutationRunner",
    "MutationReport",
]
