"""Mutation Testing Library - Runtime mutation injection for Python tests."""

from .core import (
    Mutation,
    MutationError,
    MutationInjector,
    MutationResult,
    run_mutation_tests,
)
from .config import MutationConfig, CoverageConfig
from .runner import MutationRunner, MutationReport
from .coverage import (
    CoverageReport,
    TestCoverage,
    generate_coverage_report,
    print_coverage_report,
)

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
    # Coverage
    "CoverageConfig",
    "CoverageReport",
    "TestCoverage",
    "generate_coverage_report",
    "print_coverage_report",
]
