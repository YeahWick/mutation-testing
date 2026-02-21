"""Mutation coverage reporting.

Tracks which test functions are covered by mutation checks and reports
coverage gaps. A test is "covered" if at least one mutation targets a source
function that the test exercises.

Test-to-function mapping uses naming convention by default (test_add -> add)
and can be overridden via explicit mappings in the YAML config.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


@dataclass
class TestCoverage:
    """Coverage status for a single test function."""

    test_name: str
    test_file: str
    mapped_functions: List[str]
    mutations: List[str]

    @property
    def covered(self) -> bool:
        return len(self.mutations) > 0


@dataclass
class CoverageReport:
    """Mutation coverage report across all discovered tests."""

    tests: List[TestCoverage]
    threshold: float
    total_tests: int = 0
    covered_tests: int = 0
    uncovered_tests: int = 0
    coverage_percent: float = 0.0

    def __post_init__(self):
        self.total_tests = len(self.tests)
        self.covered_tests = sum(1 for t in self.tests if t.covered)
        self.uncovered_tests = self.total_tests - self.covered_tests
        self.coverage_percent = (
            self.covered_tests / self.total_tests * 100.0 if self.total_tests else 0.0
        )

    @property
    def meets_threshold(self) -> bool:
        return self.coverage_percent >= self.threshold

    @property
    def all_covered(self) -> bool:
        return self.uncovered_tests == 0

    def to_dict(self) -> dict:
        return {
            "total_tests": self.total_tests,
            "covered_tests": self.covered_tests,
            "uncovered_tests": self.uncovered_tests,
            "coverage_percent": round(self.coverage_percent, 1),
            "threshold": self.threshold,
            "meets_threshold": self.meets_threshold,
            "tests": [
                {
                    "test_name": t.test_name,
                    "test_file": t.test_file,
                    "mapped_functions": t.mapped_functions,
                    "mutations": t.mutations,
                    "covered": t.covered,
                }
                for t in self.tests
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def discover_tests(test_paths: List[Union[str, Path]]) -> List[dict]:
    """Discover test functions by parsing Python test files.

    Finds all functions starting with ``test_`` in the given files or
    directories.

    Args:
        test_paths: List of file or directory paths to search for tests.

    Returns:
        List of dicts with ``name`` and ``file`` keys.
    """
    tests: List[dict] = []

    for path in test_paths:
        p = Path(path)
        if p.is_file() and p.suffix == ".py":
            tests.extend(_parse_test_file(p))
        elif p.is_dir():
            for py_file in sorted(p.rglob("test_*.py")):
                tests.extend(_parse_test_file(py_file))
            for py_file in sorted(p.rglob("*_test.py")):
                if py_file not in {t["file"] for t in tests}:
                    tests.extend(_parse_test_file(py_file))

    return tests


def _parse_test_file(path: Path) -> List[dict]:
    """Extract test function names from a Python file using AST."""
    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError):
        return []

    tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            tests.append({"name": node.name, "file": str(path)})
    return tests


def map_test_to_functions(
    test_name: str,
    explicit_mappings: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Map a test function name to source function names.

    Uses explicit mappings first, then falls back to naming convention
    (strips the ``test_`` prefix).

    Args:
        test_name: Name of the test function (e.g. ``test_add``).
        explicit_mappings: Optional dict mapping test names to lists of
            source function names.

    Returns:
        List of source function names this test maps to.
    """
    if explicit_mappings and test_name in explicit_mappings:
        return list(explicit_mappings[test_name])

    # Convention: test_add -> add, test_is_positive -> is_positive
    if test_name.startswith("test_"):
        return [test_name[5:]]
    return []


def generate_coverage_report(
    test_paths: List[Union[str, Path]],
    mutation_config_path: Union[str, Path],
    threshold: float = 100.0,
    test_mappings: Optional[Dict[str, List[str]]] = None,
) -> CoverageReport:
    """Generate a mutation coverage report.

    Discovers tests, maps them to source functions, and checks which have
    mutations defined in the config.

    Args:
        test_paths: Paths to test files or directories.
        mutation_config_path: Path to the mutations YAML config.
        threshold: Minimum coverage percentage required (0-100).
        test_mappings: Optional explicit test-to-function mappings.

    Returns:
        CoverageReport with per-test coverage details.
    """
    import yaml

    config_path = Path(mutation_config_path)
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Build set of functions that have mutations defined
    mutated_functions: Dict[str, List[str]] = {}
    for target in config_data.get("targets", []):
        for m in target.get("mutations", []):
            func = m.get("function", "")
            mid = m.get("id", "")
            if func:
                mutated_functions.setdefault(func, []).append(mid)

    # Discover tests
    tests = discover_tests(test_paths)

    # Build coverage entries
    coverages = []
    for test_info in tests:
        mapped = map_test_to_functions(test_info["name"], test_mappings)
        mutation_ids = []
        for func_name in mapped:
            mutation_ids.extend(mutated_functions.get(func_name, []))

        coverages.append(
            TestCoverage(
                test_name=test_info["name"],
                test_file=test_info["file"],
                mapped_functions=mapped,
                mutations=mutation_ids,
            )
        )

    return CoverageReport(tests=coverages, threshold=threshold)


def print_coverage_report(report: CoverageReport) -> None:
    """Print a formatted coverage report to stdout."""
    print()
    print("=" * 60)
    print("MUTATION COVERAGE REPORT")
    print("=" * 60)
    print()

    for test in report.tests:
        symbol = "\u2713" if test.covered else "\u2717"
        funcs = ", ".join(test.mapped_functions) if test.mapped_functions else "(none)"
        count = len(test.mutations)
        print(f"[{symbol}] {test.test_name}")
        print(f"      functions: {funcs}")
        if test.covered:
            print(f"      mutations: {count} ({', '.join(test.mutations)})")
        else:
            print(f"      mutations: 0")

    print()
    print("=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Total tests:     {report.total_tests}")
    print(f"Covered:         {report.covered_tests}")
    print(f"Uncovered:       {report.uncovered_tests}")
    print(f"Coverage:        {report.coverage_percent:.1f}%")
    print(f"Threshold:       {report.threshold:.1f}%")
    status = "PASS" if report.meets_threshold else "FAIL"
    print(f"Status:          {status}")
    print("=" * 60)

    if report.uncovered_tests > 0:
        print()
        print("UNCOVERED TESTS (add mutations for these!):")
        for t in report.tests:
            if not t.covered:
                funcs = ", ".join(t.mapped_functions) if t.mapped_functions else "(unmapped)"
                print(f"  - {t.test_name} -> {funcs}")
