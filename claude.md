# Mutation Testing Framework

## Project Overview

A Python mutation testing framework that validates test suite quality by injecting runtime mutations and verifying tests catch them.

## Core Goals

### 1. Runtime Mutation Injection
- Mutate code after import, not before
- No source file modification
- AST-based bytecode replacement at runtime
- Each test run gets fresh mutation state

### 2. Human-Readable Configuration
- Plain Python code in configuration (not AST syntax)
- Developers write mutations without AST knowledge
- Config files serve as documentation
- Easy to maintain and version control

### 3. Live Testing Integration
- Seamless pytest plugin integration
- Inject mutations during test execution
- Track which mutations are killed vs survive
- Clear reporting on test suite effectiveness

### 4. Targeted Mutations
- Config specifies exact functions/lines to mutate
- Pattern matching using AST for accuracy
- Support for common mutation operators:
  - Arithmetic operators (+ to -, * to /)
  - Comparison operators (> to >=, == to !=)
  - Boolean operators (and to or)
  - Boundary conditions
  - Return value negation

## Key Principles

1. **No Source File Modification** - Original source files are never changed
2. **AST Matching** - Parse both config and source to find matching patterns
3. **Bytecode Replacement** - Replace function code objects at runtime
4. **Isolation** - Each test run gets a fresh mutation state
5. **Reporting** - Clear reports on which mutations were caught vs survived

## Architecture

### Core Components

- **MutationInjector**: Runtime code injection using AST transformation
- **PatternReplacer**: AST node transformer for pattern matching
- **Mutation Config**: YAML-based mutation specifications
- **Test Runner**: Integration with pytest for automated testing
- **Reporter**: Results tracking and visualization

### Data Flow

1. Load mutation configuration
2. Import target modules
3. For each mutation:
   - Inject mutation into function bytecode
   - Run test suite
   - Record if mutation killed (test failed) or survived (test passed)
   - Restore original code
4. Generate mutation score report

## Project Structure

```
mutation-testing/
├── mutation_testing/       # Core framework code
│   ├── core.py            # MutationInjector and PatternReplacer
│   └── __init__.py
├── example/               # Example usage
│   ├── src/              # Sample code to mutate
│   ├── tests/            # Sample tests
│   └── run_mutations.py  # Example mutation runner
├── docs/                 # Documentation
│   └── pytest_mutation_plan.md  # Implementation plan
└── doc/
    └── ast_parsing_guide.md     # AST parsing reference
```

## Usage Example

```python
from mutation_testing.core import Mutation, run_mutation_tests

mutations = [
    Mutation(
        id="calc-001",
        function="add",
        original="return a + b",
        mutant="return a - b",
        description="Replace addition with subtraction"
    )
]

results = run_mutation_tests(
    mutations=mutations,
    test_runner=lambda: pytest.main(["-q", "tests/"]) == 0,
    module_name="calculator"
)

# Calculate mutation score
killed = sum(1 for r in results if r.killed)
score = killed / len(results)
print(f"Mutation Score: {score:.1%}")
```

## Mutation Score

The mutation score measures test suite effectiveness:

```
Mutation Score = Killed Mutations / Total Mutations
```

- **Killed**: Test suite detected the mutation (test failed) ✓
- **Survived**: Mutation went undetected (test passed) ✗

A higher score indicates better test coverage and quality.

## Current Status

**Phase 1 Complete**: Core infrastructure implemented
- ✓ AST pattern matching engine
- ✓ Runtime mutation injection
- ✓ Basic mutation runner
- ✓ Pattern replacement transformer

**Next Steps**:
- Pytest plugin integration
- YAML configuration loader
- Comprehensive reporting
- Support for more AST node types
- Parallel mutation execution

## Technical Highlights

### Why AST-Based?

1. **Accuracy**: Ignores whitespace and formatting
2. **Reliability**: Won't match comments or strings accidentally
3. **Flexibility**: Matches semantically equivalent code
4. **Safety**: Ensures valid Python syntax

### Why Runtime Injection?

1. **Speed**: No file rewriting or reimporting
2. **Safety**: Original sources remain untouched
3. **Isolation**: Clean state between mutations
4. **Parallel-safe**: Different processes can run different mutations

## Documentation

- `/docs/pytest_mutation_plan.md`: Detailed implementation plan with architecture, API reference, and examples
- `/doc/ast_parsing_guide.md`: Guide to Python AST parsing and manipulation

## Contributing

When adding new features:
1. Maintain AST-based approach for accuracy
2. Keep configuration human-readable
3. Ensure no source file modification
4. Add tests for new mutation types
5. Update documentation with examples
