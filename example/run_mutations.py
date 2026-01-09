#!/usr/bin/env python3
"""
Run mutation testing on the calculator module.

This script demonstrates how to use the mutation-testing library
to inject mutations at runtime and verify tests catch them.
"""

import sys
from pathlib import Path

# Add src to path so we can import calculator
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mutation_testing import Mutation, MutationInjector, run_mutation_tests

# Import the module we want to mutate
import calculator


def run_tests() -> bool:
    """Run all tests, return True if they pass."""
    try:
        # Test add
        assert calculator.add(2, 3) == 5
        assert calculator.add(-1, 1) == 0
        assert calculator.add(0, 0) == 0

        # Test subtract
        assert calculator.subtract(5, 3) == 2
        assert calculator.subtract(1, 1) == 0

        # Test is_positive
        assert calculator.is_positive(5) is True
        assert calculator.is_positive(-1) is False
        # Missing: is_positive(0) - mutation will survive!

        # Test clamp
        assert calculator.clamp(5, 0, 10) == 5
        assert calculator.clamp(-5, 0, 10) == 0
        assert calculator.clamp(15, 0, 10) == 10

        return True
    except AssertionError:
        return False


def main():
    print("=" * 60)
    print("MUTATION TESTING - Calculator Module")
    print("=" * 60)
    print()

    # Define mutations to test
    mutations = [
        Mutation(
            id="add-001",
            function="add",
            original="return a + b",
            mutant="return a - b",
            description="Replace + with - in add()",
        ),
        Mutation(
            id="add-002",
            function="add",
            original="return a + b",
            mutant="return a * b",
            description="Replace + with * in add()",
        ),
        Mutation(
            id="sub-001",
            function="subtract",
            original="return a - b",
            mutant="return a + b",
            description="Replace - with + in subtract()",
        ),
        Mutation(
            id="pos-001",
            function="is_positive",
            original="return x > 0",
            mutant="return x >= 0",
            description="Replace > with >= in is_positive()",
        ),
        Mutation(
            id="pos-002",
            function="is_positive",
            original="return x > 0",
            mutant="return x < 0",
            description="Replace > with < in is_positive()",
        ),
        Mutation(
            id="clamp-001",
            function="clamp",
            original="value < min_val",
            mutant="value <= min_val",
            description="Off-by-one in lower bound check",
        ),
    ]

    # Run mutation tests
    results = run_mutation_tests(
        mutations=mutations,
        test_runner=run_tests,
        module_name="calculator",
    )

    # Print results
    print()
    for result in results:
        status = "KILLED" if result.killed else "SURVIVED"
        symbol = "\u2713" if result.killed else "\u2717"
        print(f"[{symbol}] [{result.mutation.id}] {result.mutation.description}: {status}")
        if result.error:
            print(f"    Error: {result.error}")

    # Summary
    killed = sum(1 for r in results if r.killed)
    total = len(results)
    survived = total - killed
    score = killed / total if total else 0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total mutations:  {total}")
    print(f"Killed:           {killed}")
    print(f"Survived:         {survived}")
    print(f"Mutation Score:   {score:.1%}")
    print("=" * 60)

    if survived > 0:
        print()
        print("SURVIVING MUTATIONS (improve your tests!):")
        for r in results:
            if not r.killed:
                print(f"  - [{r.mutation.id}] {r.mutation.description}")

    return 0 if survived == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
