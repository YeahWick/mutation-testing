# Pytest Live Mutation Testing Framework - Implementation Plan

This document outlines the implementation plan for a pytest mutation testing framework that injects mutations at runtime after code is imported, using a human-readable configuration format.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration Format](#configuration-format)
4. [Core Components](#core-components)
5. [Implementation Phases](#implementation-phases)
6. [API Reference](#api-reference)
7. [Example Usage](#example-usage)

---

## Overview

### Goals

- **Runtime Mutation Injection**: Mutate code after it's imported, not before
- **Human-Readable Config**: Use plain Python code in configuration, not AST syntax
- **Live Testing**: Inject mutations during test execution to verify tests catch them
- **Pytest Integration**: Seamless integration with pytest as a plugin
- **Targeted Mutations**: Config specifies exactly which functions/lines to mutate

### Key Principles

1. **No Source File Modification**: Original source files are never changed
2. **AST Matching**: Parse both config and source to find matching patterns
3. **Bytecode Replacement**: Replace function code objects at runtime
4. **Isolation**: Each test run gets a fresh mutation state
5. **Reporting**: Clear reports on which mutations were caught vs. survived

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pytest Plugin Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│  pytest_configure  │  pytest_runtest_setup  │  pytest_runtest_call  │
└─────────┬───────────────────┬───────────────────────┬───────────────┘
          │                   │                       │
          ▼                   ▼                       ▼
┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Config Loader  │  │  Mutation Injector  │  │  Result Collector   │
│                 │  │                     │  │                     │
│  - Parse YAML   │  │  - AST matching     │  │  - Track kills      │
│  - Validate     │  │  - Code replacement │  │  - Track survivors  │
│  - Build plans  │  │  - Bytecode swap    │  │  - Generate report  │
└────────┬────────┘  └──────────┬──────────┘  └─────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AST Engine                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Pattern Parser  │  Code Matcher  │  AST Transformer  │  Compiler   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Configuration Loading**: Parse `mutations.yaml` to get mutation definitions
2. **Module Import Hook**: Intercept module imports to track loaded modules
3. **Pre-Test Injection**: Before each test, inject the specified mutation
4. **Test Execution**: Run the test with mutated code
5. **Result Collection**: Record if mutation was killed (test failed) or survived
6. **Post-Test Restore**: Restore original code for next mutation
7. **Report Generation**: Summarize mutation testing results

---

## Configuration Format

The configuration file uses YAML with human-readable Python code snippets.

### File: `mutations.yaml`

```yaml
# Mutation Testing Configuration
version: "1.0"

# Global settings
settings:
  timeout: 30  # seconds per mutation test
  fail_fast: false  # stop on first surviving mutation
  parallel: false  # run mutations in parallel (future)

# Target modules to mutate
targets:
  - module: "myapp.calculator"
    file: "src/myapp/calculator.py"

    mutations:
      # Mutation 1: Change addition to subtraction
      - id: "calc-001"
        function: "add"
        description: "Replace addition with subtraction"
        original: "return a + b"
        mutant: "return a - b"

      # Mutation 2: Change comparison operator
      - id: "calc-002"
        function: "is_positive"
        description: "Replace > with >="
        original: "return x > 0"
        mutant: "return x >= 0"

      # Mutation 3: Change boundary check
      - id: "calc-003"
        function: "clamp"
        line: 15  # optional: target specific line
        description: "Off-by-one error in upper bound"
        original: "if value > max_val:"
        mutant: "if value >= max_val:"

  - module: "myapp.validators"
    file: "src/myapp/validators.py"

    mutations:
      # Mutation 4: Negate boolean condition
      - id: "val-001"
        function: "validate_email"
        description: "Invert validation result"
        original: "return is_valid"
        mutant: "return not is_valid"

      # Mutation 5: Replace AND with OR
      - id: "val-002"
        function: "check_credentials"
        description: "Replace AND with OR in validation"
        original: "if username_valid and password_valid:"
        mutant: "if username_valid or password_valid:"
```

### Configuration Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Config format version |
| `settings.timeout` | int | No | Max seconds per mutation (default: 30) |
| `settings.fail_fast` | bool | No | Stop on first survivor (default: false) |
| `targets[].module` | string | Yes | Python module path |
| `targets[].file` | string | Yes | Source file path |
| `targets[].mutations[].id` | string | Yes | Unique mutation identifier |
| `targets[].mutations[].function` | string | Yes | Target function name |
| `targets[].mutations[].description` | string | No | Human-readable description |
| `targets[].mutations[].original` | string | Yes | Original Python code to match |
| `targets[].mutations[].mutant` | string | Yes | Replacement Python code |
| `targets[].mutations[].line` | int | No | Specific line number (for disambiguation) |

---

## Core Components

### 1. Configuration Loader (`config_loader.py`)

Responsible for parsing and validating mutation configuration files.

```python
# Pseudo-implementation
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class MutationSpec:
    """Specification for a single mutation."""
    id: str
    function: str
    original: str
    mutant: str
    description: Optional[str] = None
    line: Optional[int] = None

@dataclass
class TargetModule:
    """A module with mutations to apply."""
    module: str
    file: str
    mutations: List[MutationSpec]

@dataclass
class MutationConfig:
    """Complete mutation configuration."""
    version: str
    timeout: int
    fail_fast: bool
    targets: List[TargetModule]

class ConfigLoader:
    """Load and validate mutation configuration."""

    def load(self, path: str) -> MutationConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data)

    def _parse(self, data: dict) -> MutationConfig:
        """Parse raw YAML data into configuration objects."""
        # Validate and construct dataclasses
        ...

    def validate(self, config: MutationConfig) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        # Check all referenced files exist
        # Validate Python syntax in original/mutant
        # Check for duplicate IDs
        return errors
```

### 2. AST Pattern Matcher (`pattern_matcher.py`)

Matches human-readable Python code patterns against source AST.

```python
import ast
from typing import Optional, List, Tuple

class PatternMatcher:
    """Match Python code patterns in source files."""

    def __init__(self, source_code: str):
        self.source = source_code
        self.tree = ast.parse(source_code)
        self.lines = source_code.splitlines()

    def find_function(self, name: str) -> Optional[ast.FunctionDef]:
        """Find a function definition by name."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def find_pattern_in_function(
        self,
        func_name: str,
        pattern: str,
        line_hint: Optional[int] = None
    ) -> Optional[Tuple[int, int, ast.AST]]:
        """
        Find a code pattern within a function.

        Args:
            func_name: Name of function to search in
            pattern: Python code pattern to find
            line_hint: Optional line number to narrow search

        Returns:
            Tuple of (start_line, end_line, matched_node) or None
        """
        func = self.find_function(func_name)
        if not func:
            return None

        # Parse the pattern to get its AST structure
        pattern_ast = self._parse_pattern(pattern)
        if not pattern_ast:
            return None

        # Walk function body to find matching node
        for node in ast.walk(func):
            if self._ast_matches(node, pattern_ast):
                if line_hint and node.lineno != line_hint:
                    continue
                return (node.lineno, node.end_lineno, node)

        return None

    def _parse_pattern(self, pattern: str) -> Optional[ast.AST]:
        """Parse a code pattern into AST."""
        try:
            # Try as expression first
            tree = ast.parse(pattern, mode='eval')
            return tree.body
        except SyntaxError:
            pass

        try:
            # Try as statement
            tree = ast.parse(pattern, mode='exec')
            if tree.body:
                return tree.body[0]
        except SyntaxError:
            pass

        return None

    def _ast_matches(self, source_node: ast.AST, pattern_node: ast.AST) -> bool:
        """
        Check if source AST node matches pattern AST node.

        This performs structural matching, ignoring:
        - Line numbers and column offsets
        - Context (Load/Store/Del)
        - Exact whitespace
        """
        if type(source_node) != type(pattern_node):
            return False

        # Compare relevant fields based on node type
        if isinstance(pattern_node, ast.Name):
            return source_node.id == pattern_node.id

        if isinstance(pattern_node, ast.Constant):
            return source_node.value == pattern_node.value

        if isinstance(pattern_node, ast.BinOp):
            return (
                type(source_node.op) == type(pattern_node.op) and
                self._ast_matches(source_node.left, pattern_node.left) and
                self._ast_matches(source_node.right, pattern_node.right)
            )

        if isinstance(pattern_node, ast.Compare):
            if len(source_node.ops) != len(pattern_node.ops):
                return False
            if not all(type(a) == type(b) for a, b in
                       zip(source_node.ops, pattern_node.ops)):
                return False
            if not self._ast_matches(source_node.left, pattern_node.left):
                return False
            return all(self._ast_matches(a, b) for a, b in
                      zip(source_node.comparators, pattern_node.comparators))

        if isinstance(pattern_node, ast.Return):
            if pattern_node.value is None:
                return source_node.value is None
            return self._ast_matches(source_node.value, pattern_node.value)

        if isinstance(pattern_node, ast.If):
            return self._ast_matches(source_node.test, pattern_node.test)

        # Add more node types as needed
        return False
```

### 3. Mutation Injector (`injector.py`)

Performs runtime injection of mutations into loaded modules.

```python
import ast
import sys
import types
from typing import Callable, Optional

class MutationInjector:
    """Inject mutations into running Python code."""

    def __init__(self):
        self._original_code: dict = {}  # Store original code objects

    def inject(
        self,
        module_name: str,
        function_name: str,
        original_pattern: str,
        mutant_pattern: str,
        line_hint: Optional[int] = None
    ) -> bool:
        """
        Inject a mutation into a loaded module's function.

        Args:
            module_name: Fully qualified module name
            function_name: Name of function to mutate
            original_pattern: Code pattern to find
            mutant_pattern: Code to replace with
            line_hint: Optional line number for disambiguation

        Returns:
            True if mutation was successfully injected
        """
        # Get the module
        module = sys.modules.get(module_name)
        if not module:
            return False

        # Get the function
        func = getattr(module, function_name, None)
        if not func or not callable(func):
            return False

        # Store original if not already stored
        key = f"{module_name}.{function_name}"
        if key not in self._original_code:
            self._original_code[key] = func.__code__

        # Get source and create mutated version
        import inspect
        try:
            source = inspect.getsource(func)
        except OSError:
            return False

        mutated_source = self._apply_mutation(
            source, original_pattern, mutant_pattern, line_hint
        )
        if not mutated_source:
            return False

        # Compile and inject
        new_code = self._compile_function(mutated_source, function_name)
        if not new_code:
            return False

        # Replace the function's code object
        func.__code__ = new_code
        return True

    def restore(self, module_name: str, function_name: str) -> bool:
        """Restore original function code."""
        key = f"{module_name}.{function_name}"
        if key not in self._original_code:
            return False

        module = sys.modules.get(module_name)
        if not module:
            return False

        func = getattr(module, function_name, None)
        if not func:
            return False

        func.__code__ = self._original_code[key]
        return True

    def restore_all(self):
        """Restore all mutated functions to original."""
        for key in list(self._original_code.keys()):
            module_name, function_name = key.rsplit('.', 1)
            self.restore(module_name, function_name)
        self._original_code.clear()

    def _apply_mutation(
        self,
        source: str,
        original: str,
        mutant: str,
        line_hint: Optional[int]
    ) -> Optional[str]:
        """Apply mutation to source code using AST."""
        tree = ast.parse(source)

        # Find the node matching original pattern
        original_ast = self._parse_pattern(original)
        mutant_ast = self._parse_pattern(mutant)

        if not original_ast or not mutant_ast:
            return None

        # Transform the tree
        transformer = PatternReplacer(original_ast, mutant_ast, line_hint)
        new_tree = transformer.visit(tree)

        if not transformer.replaced:
            return None

        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def _parse_pattern(self, pattern: str) -> Optional[ast.AST]:
        """Parse pattern string to AST node."""
        try:
            tree = ast.parse(pattern, mode='eval')
            return tree.body
        except SyntaxError:
            try:
                tree = ast.parse(pattern, mode='exec')
                return tree.body[0] if tree.body else None
            except SyntaxError:
                return None

    def _compile_function(
        self,
        source: str,
        func_name: str
    ) -> Optional[types.CodeType]:
        """Compile source and extract function's code object."""
        try:
            code = compile(source, '<mutation>', 'exec')
            # Execute to get the function
            namespace = {}
            exec(code, namespace)
            func = namespace.get(func_name)
            return func.__code__ if func else None
        except Exception:
            return None


class PatternReplacer(ast.NodeTransformer):
    """AST transformer that replaces pattern matches."""

    def __init__(
        self,
        original: ast.AST,
        replacement: ast.AST,
        line_hint: Optional[int] = None
    ):
        self.original = original
        self.replacement = replacement
        self.line_hint = line_hint
        self.replaced = False

    def visit(self, node):
        """Visit node and replace if matches pattern."""
        if self._matches(node, self.original):
            if self.line_hint is None or node.lineno == self.line_hint:
                self.replaced = True
                return ast.copy_location(
                    self._deep_copy(self.replacement),
                    node
                )
        return self.generic_visit(node)

    def _matches(self, node: ast.AST, pattern: ast.AST) -> bool:
        """Check if node matches pattern (structural comparison)."""
        # Implementation similar to PatternMatcher._ast_matches
        ...

    def _deep_copy(self, node: ast.AST) -> ast.AST:
        """Deep copy an AST node."""
        import copy
        return copy.deepcopy(node)
```

### 4. Pytest Plugin (`plugin.py`)

Integrates the mutation framework with pytest.

```python
import pytest
from typing import List, Optional
from .config_loader import ConfigLoader, MutationConfig, MutationSpec
from .injector import MutationInjector

class MutationTestingPlugin:
    """Pytest plugin for mutation testing."""

    def __init__(self, config: MutationConfig):
        self.config = config
        self.injector = MutationInjector()
        self.results: List[MutationResult] = []
        self.current_mutation: Optional[MutationSpec] = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config):
        """Initialize mutation testing."""
        # Validate configuration
        # Pre-load target modules
        pass

    @pytest.hookimpl
    def pytest_collection_modifyitems(self, session, config, items):
        """Modify test collection for mutation testing."""
        # For each mutation, we'll run all tests
        # This multiplies tests by number of mutations
        pass

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        """Inject mutation before each test."""
        mutation_id = self._get_mutation_id(item)
        if mutation_id:
            mutation = self._find_mutation(mutation_id)
            if mutation:
                self.current_mutation = mutation
                self._inject_mutation(mutation)
        yield

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        """Track test outcome for mutation."""
        outcome = yield

        if self.current_mutation:
            # Test passed = mutation survived (bad)
            # Test failed = mutation killed (good)
            killed = outcome.excinfo is not None
            self.results.append(MutationResult(
                mutation=self.current_mutation,
                killed=killed,
                test_name=item.name
            ))

    @pytest.hookimpl
    def pytest_runtest_teardown(self, item):
        """Restore original code after test."""
        if self.current_mutation:
            self.injector.restore_all()
            self.current_mutation = None

    @pytest.hookimpl
    def pytest_terminal_summary(self, terminalreporter):
        """Print mutation testing summary."""
        self._print_report(terminalreporter)

    def _inject_mutation(self, mutation: MutationSpec):
        """Inject a specific mutation."""
        target = self._find_target_for_mutation(mutation)
        self.injector.inject(
            module_name=target.module,
            function_name=mutation.function,
            original_pattern=mutation.original,
            mutant_pattern=mutation.mutant,
            line_hint=mutation.line
        )

    def _print_report(self, reporter):
        """Print mutation testing results."""
        reporter.write_sep("=", "MUTATION TESTING RESULTS")

        killed = sum(1 for r in self.results if r.killed)
        total = len(self.results)
        score = killed / total if total > 0 else 0

        reporter.write_line(f"Total mutations: {total}")
        reporter.write_line(f"Killed: {killed}")
        reporter.write_line(f"Survived: {total - killed}")
        reporter.write_line(f"Mutation Score: {score:.1%}")

        if total - killed > 0:
            reporter.write_sep("-", "SURVIVING MUTATIONS")
            for result in self.results:
                if not result.killed:
                    reporter.write_line(
                        f"  [{result.mutation.id}] {result.mutation.description}"
                    )


def pytest_addoption(parser):
    """Add command line options."""
    group = parser.getgroup('mutation')
    group.addoption(
        '--mutation-config',
        action='store',
        default='mutations.yaml',
        help='Path to mutation configuration file'
    )
    group.addoption(
        '--mutation-id',
        action='store',
        default=None,
        help='Run only specific mutation ID'
    )
    group.addoption(
        '--mutation-testing',
        action='store_true',
        default=False,
        help='Enable mutation testing mode'
    )


def pytest_configure(config):
    """Configure mutation testing plugin."""
    if config.option.mutation_testing:
        config_path = config.option.mutation_config
        loader = ConfigLoader()
        mutation_config = loader.load(config_path)

        plugin = MutationTestingPlugin(mutation_config)
        config.pluginmanager.register(plugin, 'mutation_testing')
```

### 5. Result Reporting (`reporter.py`)

Generate detailed mutation testing reports.

```python
from dataclasses import dataclass
from typing import List
import json
from datetime import datetime

@dataclass
class MutationResult:
    """Result of a single mutation test."""
    mutation_id: str
    function: str
    description: str
    original: str
    mutant: str
    killed: bool
    test_name: str
    duration: float = 0.0

class MutationReporter:
    """Generate mutation testing reports."""

    def __init__(self, results: List[MutationResult]):
        self.results = results

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def killed(self) -> int:
        return sum(1 for r in self.results if r.killed)

    @property
    def survived(self) -> int:
        return self.total - self.killed

    @property
    def score(self) -> float:
        return self.killed / self.total if self.total > 0 else 0.0

    def console_report(self) -> str:
        """Generate console-friendly report."""
        lines = [
            "=" * 60,
            "MUTATION TESTING REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "SUMMARY",
            "-" * 60,
            f"Total Mutations:  {self.total}",
            f"Killed:           {self.killed}",
            f"Survived:         {self.survived}",
            f"Mutation Score:   {self.score:.1%}",
            "",
        ]

        if self.survived > 0:
            lines.extend([
                "SURVIVING MUTATIONS (tests should catch these!)",
                "-" * 60,
            ])
            for result in self.results:
                if not result.killed:
                    lines.extend([
                        f"[{result.mutation_id}] {result.function}",
                        f"  Description: {result.description}",
                        f"  Original:    {result.original}",
                        f"  Mutant:      {result.mutant}",
                        "",
                    ])

        lines.extend([
            "=" * 60,
        ])

        return "\n".join(lines)

    def json_report(self) -> str:
        """Generate JSON report."""
        return json.dumps({
            "generated": datetime.now().isoformat(),
            "summary": {
                "total": self.total,
                "killed": self.killed,
                "survived": self.survived,
                "score": self.score
            },
            "mutations": [
                {
                    "id": r.mutation_id,
                    "function": r.function,
                    "description": r.description,
                    "original": r.original,
                    "mutant": r.mutant,
                    "killed": r.killed,
                    "test": r.test_name,
                    "duration": r.duration
                }
                for r in self.results
            ]
        }, indent=2)

    def html_report(self) -> str:
        """Generate HTML report."""
        # HTML template with results
        ...
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)

**Goal**: Establish the foundational components for AST pattern matching and code injection.

**Deliverables**:
1. `config_loader.py` - YAML configuration parser with validation
2. `pattern_matcher.py` - AST-based pattern matching engine
3. `exceptions.py` - Custom exception hierarchy

**Tasks**:
- [ ] Define dataclasses for configuration schema
- [ ] Implement YAML parsing with schema validation
- [ ] Build AST pattern parser for human-readable code
- [ ] Create structural AST comparison algorithm
- [ ] Add support for common node types (Return, BinOp, Compare, If, BoolOp)
- [ ] Write unit tests for pattern matching

### Phase 2: Mutation Injection Engine

**Goal**: Implement runtime code injection and restoration.

**Deliverables**:
1. `injector.py` - Runtime mutation injector
2. `transformer.py` - AST transformation utilities

**Tasks**:
- [ ] Implement code object extraction from functions
- [ ] Build AST node replacement transformer
- [ ] Create function code object replacement mechanism
- [ ] Implement original code storage and restoration
- [ ] Handle edge cases (closures, decorators, class methods)
- [ ] Add timeout protection for mutations
- [ ] Write integration tests for injection

### Phase 3: Pytest Plugin

**Goal**: Create seamless pytest integration.

**Deliverables**:
1. `plugin.py` - Pytest plugin with hooks
2. `conftest.py` - Example configuration

**Tasks**:
- [ ] Implement pytest command-line options
- [ ] Create test collection modification for mutation runs
- [ ] Build pre-test mutation injection hook
- [ ] Implement post-test restoration hook
- [ ] Add mutation result tracking
- [ ] Create terminal summary reporter
- [ ] Support for running specific mutation IDs
- [ ] Write plugin integration tests

### Phase 4: Reporting and Polish

**Goal**: Comprehensive reporting and user experience.

**Deliverables**:
1. `reporter.py` - Report generation
2. Documentation and examples

**Tasks**:
- [ ] Console report formatter
- [ ] JSON report output
- [ ] HTML report with visual indicators
- [ ] JUnit XML format for CI integration
- [ ] Surviving mutation highlighting
- [ ] Performance metrics collection
- [ ] Documentation and usage guides
- [ ] Example projects

### Phase 5: Advanced Features (Future)

**Goal**: Enhanced capabilities for production use.

**Potential Features**:
- [ ] Parallel mutation execution
- [ ] Incremental mutation testing (only changed files)
- [ ] Auto-generation of mutation configs
- [ ] IDE integration (VS Code extension)
- [ ] Coverage-guided mutation selection
- [ ] Mutation clustering and equivalence detection

---

## API Reference

### Command Line Interface

```bash
# Run mutation testing
pytest --mutation-testing

# Use custom config file
pytest --mutation-testing --mutation-config=custom_mutations.yaml

# Run specific mutation
pytest --mutation-testing --mutation-id=calc-001

# Generate HTML report
pytest --mutation-testing --mutation-report=html

# Fail if score below threshold
pytest --mutation-testing --mutation-threshold=80
```

### Python API

```python
from pytest_mutations import MutationConfig, MutationInjector, run_mutations

# Load configuration
config = MutationConfig.from_yaml("mutations.yaml")

# Manual injection (for advanced use)
injector = MutationInjector()
injector.inject(
    module_name="myapp.math",
    function_name="add",
    original_pattern="return a + b",
    mutant_pattern="return a - b"
)

# Run your tests...

# Restore
injector.restore_all()
```

### Pytest Fixtures

```python
import pytest

@pytest.fixture
def mutation_injector():
    """Provide mutation injector for custom mutation tests."""
    from pytest_mutations import MutationInjector
    injector = MutationInjector()
    yield injector
    injector.restore_all()

def test_custom_mutation(mutation_injector):
    """Test with custom inline mutation."""
    import myapp.calculator

    mutation_injector.inject(
        "myapp.calculator",
        "add",
        "return a + b",
        "return a - b"
    )

    # This should fail, proving the test catches the mutation
    with pytest.raises(AssertionError):
        assert myapp.calculator.add(2, 3) == 5
```

---

## Example Usage

### Project Structure

```
myproject/
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── calculator.py
│       └── validators.py
├── tests/
│   ├── conftest.py
│   ├── test_calculator.py
│   └── test_validators.py
├── mutations.yaml
└── pyproject.toml
```

### Source Code (`src/myapp/calculator.py`)

```python
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

def is_positive(x: int) -> bool:
    """Check if number is positive."""
    return x > 0

def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value between min and max."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value
```

### Test Suite (`tests/test_calculator.py`)

```python
import pytest
from myapp.calculator import add, subtract, is_positive, clamp

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0

def test_is_positive():
    assert is_positive(5) == True
    assert is_positive(-1) == False
    assert is_positive(0) == False  # Edge case!

def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert clamp(10, 0, 10) == 10  # Boundary test
```

### Mutation Configuration (`mutations.yaml`)

```yaml
version: "1.0"

settings:
  timeout: 30
  fail_fast: false

targets:
  - module: "myapp.calculator"
    file: "src/myapp/calculator.py"

    mutations:
      - id: "calc-add-001"
        function: "add"
        description: "Replace + with -"
        original: "return a + b"
        mutant: "return a - b"

      - id: "calc-add-002"
        function: "add"
        description: "Replace + with *"
        original: "return a + b"
        mutant: "return a * b"

      - id: "calc-pos-001"
        function: "is_positive"
        description: "Replace > with >="
        original: "return x > 0"
        mutant: "return x >= 0"

      - id: "calc-pos-002"
        function: "is_positive"
        description: "Replace > with <"
        original: "return x > 0"
        mutant: "return x < 0"

      - id: "calc-clamp-001"
        function: "clamp"
        description: "Off-by-one in lower bound"
        original: "if value < min_val:"
        mutant: "if value <= min_val:"

      - id: "calc-clamp-002"
        function: "clamp"
        description: "Off-by-one in upper bound"
        original: "if value > max_val:"
        mutant: "if value >= max_val:"
```

### Running Mutation Tests

```bash
# Run all mutation tests
$ pytest --mutation-testing

================================ MUTATION TESTING ================================
Running 6 mutations against test suite...

[1/6] calc-add-001: Replace + with - ........................... KILLED
[2/6] calc-add-002: Replace + with * ........................... KILLED
[3/6] calc-pos-001: Replace > with >= .......................... SURVIVED
[4/6] calc-pos-002: Replace > with < ........................... KILLED
[5/6] calc-clamp-001: Off-by-one in lower bound ................ KILLED
[6/6] calc-clamp-002: Off-by-one in upper bound ................ SURVIVED

================================ MUTATION RESULTS ================================
Total Mutations:  6
Killed:           4
Survived:         2
Mutation Score:   66.7%

SURVIVING MUTATIONS (improve your tests!):
  [calc-pos-001] is_positive: Replace > with >=
    Your test doesn't verify that is_positive(0) returns False

  [calc-clamp-002] clamp: Off-by-one in upper bound
    Your test doesn't catch when clamp(10, 0, 10) returns 10 vs max_val

=================================================================================
```

### Improving Tests to Kill Survivors

```python
def test_is_positive_edge_cases():
    """Kill mutation calc-pos-001."""
    assert is_positive(0) == False  # 0 is NOT positive!
    assert is_positive(1) == True   # Smallest positive integer

def test_clamp_boundaries():
    """Kill mutation calc-clamp-002."""
    # Test exact boundary - should return the value, not max_val
    assert clamp(10, 0, 10) == 10
    # Test just over boundary
    assert clamp(11, 0, 10) == 10
```

---

## File Structure

```
pytest_mutations/
├── __init__.py
├── config_loader.py      # Configuration parsing
├── pattern_matcher.py    # AST pattern matching
├── transformer.py        # AST transformations
├── injector.py          # Runtime injection
├── plugin.py            # Pytest plugin
├── reporter.py          # Report generation
├── exceptions.py        # Custom exceptions
└── cli.py               # Command-line interface

tests/
├── conftest.py
├── test_config_loader.py
├── test_pattern_matcher.py
├── test_injector.py
├── test_plugin.py
└── fixtures/
    ├── sample_module.py
    └── sample_mutations.yaml
```

---

## Design Decisions

### Why Runtime Injection vs. Source Modification?

1. **No file system side effects** - Original sources remain untouched
2. **Speed** - No need to rewrite and reimport modules
3. **Isolation** - Each test gets a fresh state
4. **Parallel-safe** - Different processes can run different mutations

### Why Human-Readable Config vs. AST?

1. **Accessibility** - Developers can write mutations without AST knowledge
2. **Readability** - Config files serve as documentation
3. **Maintainability** - Easy to update when code changes
4. **Version Control** - Clean diffs in configuration changes

### Why AST Matching vs. Text Matching?

1. **Accuracy** - Ignores whitespace and formatting differences
2. **Reliability** - Won't match comments or strings accidentally
3. **Flexibility** - Can match semantically equivalent code
4. **Safety** - Ensures valid Python syntax in mutations
