#!/usr/bin/env python3
"""
Example: Run mutation testing on the calculator module.

Shows two ways to use the mutation testing library:
1. From config file (simplest)
2. Explicitly defining mutations
"""

import sys
from pathlib import Path

# Add src to path so we can import calculator
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mutation_testing import MutationRunner, Mutation
import calculator


def run_tests() -> bool:
    """Run all tests, return True if they pass."""
    try:
        assert calculator.add(2, 3) == 5
        assert calculator.add(-1, 1) == 0
        assert calculator.subtract(5, 3) == 2
        assert calculator.is_positive(5) is True
        assert calculator.is_positive(-1) is False
        assert calculator.clamp(5, 0, 10) == 5
        assert calculator.clamp(-5, 0, 10) == 0
        assert calculator.clamp(15, 0, 10) == 10
        return True
    except AssertionError:
        return False


def main():
    runner = MutationRunner(run_tests)

    # Option 1: Run from config file
    config_path = Path(__file__).parent / "mutations.yaml"
    report = runner.run_from_config(config_path)

    # Option 2: Run explicitly
    # report = runner.run(
    #     mutations=[
    #         Mutation(id="add-001", function="add",
    #                  original="return a + b", mutant="return a - b",
    #                  description="Replace + with -"),
    #     ],
    #     module_name="calculator",
    # )

    return 0 if report.all_killed else 1


if __name__ == "__main__":
    sys.exit(main())
