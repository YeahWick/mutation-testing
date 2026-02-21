"""High-level mutation testing runner with formatted output."""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from dataclasses import dataclass

from .core import Mutation, MutationResult, run_mutation_tests
from .config import MutationConfig
from .coverage import (
    CoverageReport,
    generate_coverage_report,
    print_coverage_report,
)


@dataclass
class MutationReport:
    """Summary report of mutation testing results."""

    results: List[MutationResult]
    total: int
    killed: int
    survived: int
    score: float

    @property
    def all_killed(self) -> bool:
        return self.survived == 0


class MutationRunner:
    """High-level interface for running mutation tests."""

    def __init__(self, test_runner: Callable[[], bool], verbose: bool = True):
        """
        Initialize the mutation runner.

        Args:
            test_runner: Function that runs tests, returns True if tests pass
            verbose: Whether to print output during execution
        """
        self.test_runner = test_runner
        self.verbose = verbose

    def run_from_config(
        self,
        config_path: Union[str, Path],
    ) -> MutationReport:
        """
        Run mutation tests from a YAML config file.

        Args:
            config_path: Path to the mutations.yaml config file

        Returns:
            MutationReport with results
        """
        config = MutationConfig.from_yaml(config_path)
        return self.run(config.mutations, config.module)

    def run(
        self,
        mutations: List[Mutation],
        module_name: str,
    ) -> MutationReport:
        """
        Run mutation tests explicitly.

        Args:
            mutations: List of Mutation objects to test
            module_name: Name of the module to mutate

        Returns:
            MutationReport with results
        """
        if self.verbose:
            self._print_header()

        results = run_mutation_tests(
            mutations=mutations,
            test_runner=self.test_runner,
            module_name=module_name,
        )

        report = self._build_report(results)

        if self.verbose:
            self._print_results(results)
            self._print_summary(report)

        return report

    def _build_report(self, results: List[MutationResult]) -> MutationReport:
        total = len(results)
        killed = sum(1 for r in results if r.killed)
        survived = total - killed
        score = killed / total if total else 0.0
        return MutationReport(
            results=results,
            total=total,
            killed=killed,
            survived=survived,
            score=score,
        )

    def _print_header(self):
        print("=" * 60)
        print("MUTATION TESTING")
        print("=" * 60)
        print()

    def _print_results(self, results: List[MutationResult]):
        print()
        for result in results:
            status = "KILLED" if result.killed else "SURVIVED"
            symbol = "\u2713" if result.killed else "\u2717"
            print(f"[{symbol}] [{result.mutation.id}] {result.mutation.description}: {status}")
            if result.error:
                print(f"    Error: {result.error}")

    def coverage_from_config(
        self,
        config_path: Union[str, Path],
        test_paths: Optional[List[Union[str, Path]]] = None,
        threshold: Optional[float] = None,
        test_mappings: Optional[Dict[str, List[str]]] = None,
    ) -> CoverageReport:
        """Generate a mutation coverage report from a config file.

        Args:
            config_path: Path to the mutations YAML config.
            test_paths: Paths to test files/directories. Overrides the
                config's ``coverage.test_paths`` if provided.
            threshold: Minimum coverage percentage (0-100). Overrides
                the config's ``coverage.threshold`` if provided.
            test_mappings: Explicit test-to-function mappings. Merged
                with the config's ``coverage.test_mappings``.

        Returns:
            CoverageReport with per-test coverage details.
        """
        config = MutationConfig.from_yaml(config_path)

        cov = config.coverage
        effective_threshold = threshold if threshold is not None else (
            cov.threshold if cov else 100.0
        )
        effective_paths = test_paths or (cov.test_paths if cov else [])
        effective_mappings = dict(cov.test_mappings) if cov else {}
        if test_mappings:
            effective_mappings.update(test_mappings)

        if not effective_paths:
            raise ValueError(
                "No test paths provided. Pass test_paths or set "
                "coverage.test_paths in the YAML config."
            )

        report = generate_coverage_report(
            test_paths=effective_paths,
            mutation_config_path=config_path,
            threshold=effective_threshold,
            test_mappings=effective_mappings or None,
        )

        if self.verbose:
            print_coverage_report(report)

        return report

    def _print_summary(self, report: MutationReport):
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total mutations:  {report.total}")
        print(f"Killed:           {report.killed}")
        print(f"Survived:         {report.survived}")
        print(f"Mutation Score:   {report.score:.1%}")
        print("=" * 60)

        if report.survived > 0:
            print()
            print("SURVIVING MUTATIONS (improve your tests!):")
            for r in report.results:
                if not r.killed:
                    print(f"  - [{r.mutation.id}] {r.mutation.description}")
