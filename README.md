# mutation-testing

A Python mutation testing framework that validates test suite quality by injecting runtime mutations and checking whether your tests catch them.

Mutations are applied at runtime using AST pattern matching — source files are never modified.

## How it works

1. You define **mutations**: intentional code changes like replacing `+` with `-` or `>` with `>=`
2. The framework **injects** each mutation into the running code by swapping the function's `__code__` object
3. Your **test suite runs** against the mutated code
4. If a test fails, the mutation was **killed** (good — your tests caught the bug)
5. If all tests pass, the mutation **survived** (bad — your tests have a gap)

The **mutation score** is the percentage of mutations killed. A higher score means stronger tests.

## Installation

Requires Python 3.10+.

```bash
pip install -e .

# With dev dependencies (pytest)
pip install -e ".[dev]"
```

## Quick start

### 1. Define mutations in YAML

Create a `mutations.yaml` file:

```yaml
version: "1.0"

settings:
  timeout: 30

targets:
  - module: "calculator"
    file: "src/calculator.py"
    mutations:
      - id: "add-001"
        function: "add"
        description: "Replace + with -"
        original: "return a + b"
        mutant: "return a - b"

      - id: "pos-001"
        function: "is_positive"
        description: "Replace > with >="
        original: "return x > 0"
        mutant: "return x >= 0"
```

### 2. Write a test runner

```python
from mutation_testing import MutationRunner

def run_tests() -> bool:
    """Return True if all tests pass."""
    try:
        assert calculator.add(2, 3) == 5
        assert calculator.add(-1, 1) == 0
        assert calculator.is_positive(5) is True
        assert calculator.is_positive(-1) is False
        return True
    except AssertionError:
        return False

runner = MutationRunner(run_tests)
report = runner.run_from_config("mutations.yaml")
```

### 3. Run it

```bash
cd example
python run_mutations.py
```

Output:

```
============================================================
MUTATION TESTING
============================================================

[✓] [add-001] Replace + with -: KILLED
[✓] [add-002] Replace + with *: KILLED
[✓] [sub-001] Replace - with +: KILLED
[✗] [pos-001] Replace > with >=: SURVIVED
[✓] [pos-002] Replace > with <: KILLED
[✓] [clamp-001] Off-by-one in lower bound: KILLED
[✗] [clamp-002] Off-by-one in upper bound: SURVIVED

============================================================
SUMMARY
============================================================
Total mutations:  7
Killed:           5
Survived:         2
Mutation Score:   71.4%
============================================================

SURVIVING MUTATIONS (improve your tests!):
  - [pos-001] Replace > with >=
  - [clamp-002] Off-by-one in upper bound
```

The surviving mutations tell you exactly where your tests are weak — in this case, missing boundary checks for `is_positive(0)` and `clamp` at its upper bound.

## API

### Defining mutations explicitly

Instead of YAML, you can define mutations in code:

```python
from mutation_testing import MutationRunner, Mutation

runner = MutationRunner(run_tests)
report = runner.run(
    mutations=[
        Mutation(
            id="add-001",
            function="add",
            original="return a + b",
            mutant="return a - b",
            description="Replace + with -",
        ),
    ],
    module_name="calculator",
)

print(f"Score: {report.score:.1%}")
print(f"All killed: {report.all_killed}")
```

### Core classes

| Class | Purpose |
|---|---|
| `Mutation` | Defines a single mutation (id, function, original pattern, mutant pattern) |
| `MutationRunner` | High-level runner — accepts a test function, runs mutations, prints results |
| `MutationReport` | Results summary with `total`, `killed`, `survived`, `score`, `all_killed` |
| `MutationConfig` | Loads mutation definitions from a YAML file |
| `MutationInjector` | Low-level engine that injects/restores mutations at runtime |
| `MutationResult` | Result of a single mutation (mutation + killed boolean) |

### Low-level API

For direct control over injection:

```python
from mutation_testing import MutationInjector

injector = MutationInjector()

# Inject a mutation
injector.inject("calculator", "add", "return a + b", "return a - b")

# Run your tests here...

# Restore the original
injector.restore("calculator", "add")

# Or restore everything at once
injector.restore_all()
```

## Supported mutation patterns

The AST pattern matcher supports any valid Python expression or statement:

- **Arithmetic operators**: `+` ↔ `-`, `*`, `/`, `//`
- **Comparison operators**: `>` ↔ `>=`, `<` ↔ `<=`, `==` ↔ `!=`
- **Boolean operators**: `and` ↔ `or`
- **Return values**: `return x` → `return None`, `return -x`
- **Any expression** parseable as Python

Patterns are matched structurally via AST, so whitespace and formatting differences are ignored.

## Project structure

```
mutation_testing/
├── __init__.py     # Public API exports
├── core.py         # AST pattern matching, injection engine, Mutation/MutationResult
├── config.py       # YAML configuration loader
└── runner.py       # MutationRunner and MutationReport

example/
├── src/calculator.py       # Sample module under test
├── tests/test_calculator.py  # Sample test suite
├── mutations.yaml          # Sample mutation config
└── run_mutations.py        # Example runner script
```

## Key concepts

| Term | Meaning |
|---|---|
| **Mutation** | An intentional code change (e.g., `+` → `-`) |
| **Killed** | Tests detected the mutation (test failed) |
| **Survived** | Tests passed despite the mutation — tests need improvement |
| **Mutation score** | Killed / Total (higher is better) |

## License

MIT
