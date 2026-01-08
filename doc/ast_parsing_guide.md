# Python AST Parsing Guide

This guide covers how to parse Python code, match patterns, and replace code using Python's Abstract Syntax Tree (AST) module. These techniques are fundamental to mutation testing implementations.

## Table of Contents

1. [Introduction to AST](#introduction-to-ast)
2. [Parsing Python Code](#parsing-python-code)
3. [Understanding the AST Structure](#understanding-the-ast-structure)
4. [Matching Code Patterns](#matching-code-patterns)
5. [Replacing Code (Mutations)](#replacing-code-mutations)
6. [Converting AST Back to Code](#converting-ast-back-to-code)
7. [Practical Examples](#practical-examples)
8. [Complete Mutation Testing Example](#complete-mutation-testing-example)

---

## Introduction to AST

The `ast` module in Python provides tools for working with Abstract Syntax Trees. An AST represents the syntactic structure of Python code as a tree, where each node represents a construct (like expressions, statements, operators).

```python
import ast
```

**Key benefits of AST parsing:**
- Understands code structure semantically, not just textually
- Maintains valid Python syntax
- Provides precise node location information
- Enables safe code transformations

---

## Parsing Python Code

### Basic Parsing

```python
import ast

# Parse a string of Python code
source_code = """
def add(a, b):
    return a + b
"""

tree = ast.parse(source_code)
print(ast.dump(tree, indent=2))
```

**Output:**
```
Module(
  body=[
    FunctionDef(
      name='add',
      args=arguments(
        posonlyargs=[],
        args=[
          arg(arg='a'),
          arg(arg='b')],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Return(
          value=BinOp(
            left=Name(id='a', ctx=Load()),
            op=Add(),
            right=Name(id='b', ctx=Load())))],
      decorator_list=[])],
  type_ignores=[])
```

### Parsing from a File

```python
import ast

def parse_file(filepath):
    """Parse a Python file and return its AST."""
    with open(filepath, 'r') as f:
        source = f.read()
    return ast.parse(source, filename=filepath)

tree = parse_file('my_module.py')
```

### Parsing Single Expressions

```python
# Parse a single expression (not a module)
expr = ast.parse("x + y * 2", mode='eval')
print(ast.dump(expr, indent=2))
```

**Output:**
```
Expression(
  body=BinOp(
    left=Name(id='x', ctx=Load()),
    op=Add(),
    right=BinOp(
      left=Name(id='y', ctx=Load()),
      op=Mult(),
      right=Constant(value=2))))
```

---

## Understanding the AST Structure

### Common Node Types

| Node Type | Description | Example Code |
|-----------|-------------|--------------|
| `Module` | Top-level container | Entire file |
| `FunctionDef` | Function definition | `def foo():` |
| `ClassDef` | Class definition | `class Foo:` |
| `Return` | Return statement | `return x` |
| `Assign` | Assignment | `x = 5` |
| `BinOp` | Binary operation | `a + b` |
| `Compare` | Comparison | `a > b` |
| `BoolOp` | Boolean operation | `a and b` |
| `UnaryOp` | Unary operation | `-x`, `not x` |
| `If` | If statement | `if x:` |
| `For` | For loop | `for i in x:` |
| `While` | While loop | `while x:` |
| `Call` | Function call | `foo(x)` |
| `Name` | Variable name | `x` |
| `Constant` | Literal value | `5`, `"hello"` |

### Operators

**Binary Operators (`ast.operator`):**
```python
ast.Add      # +
ast.Sub      # -
ast.Mult     # *
ast.Div      # /
ast.FloorDiv # //
ast.Mod      # %
ast.Pow      # **
ast.BitOr    # |
ast.BitXor   # ^
ast.BitAnd   # &
ast.LShift   # <<
ast.RShift   # >>
```

**Comparison Operators (`ast.cmpop`):**
```python
ast.Eq       # ==
ast.NotEq    # !=
ast.Lt       # <
ast.LtE      # <=
ast.Gt       # >
ast.GtE      # >=
ast.Is       # is
ast.IsNot    # is not
ast.In       # in
ast.NotIn    # not in
```

**Boolean Operators (`ast.boolop`):**
```python
ast.And      # and
ast.Or       # or
```

**Unary Operators (`ast.unaryop`):**
```python
ast.UAdd     # +x
ast.USub     # -x
ast.Not      # not x
ast.Invert   # ~x
```

### Node Attributes

Every AST node has useful attributes:

```python
import ast

source = """x = 1 + 2"""
tree = ast.parse(source)

# Walk through all nodes
for node in ast.walk(tree):
    if hasattr(node, 'lineno'):
        print(f"{node.__class__.__name__} at line {node.lineno}, col {node.col_offset}")
```

**Key attributes:**
- `lineno` - Line number (1-indexed)
- `col_offset` - Column offset (0-indexed)
- `end_lineno` - End line number
- `end_col_offset` - End column offset

---

## Matching Code Patterns

### Using ast.walk()

Walk through all nodes in the tree:

```python
import ast

source = """
def calculate(x, y):
    result = x + y
    if result > 10:
        return result * 2
    return result
"""

tree = ast.parse(source)

# Find all binary operations
for node in ast.walk(tree):
    if isinstance(node, ast.BinOp):
        print(f"Found BinOp: {ast.dump(node)}")
```

### Using ast.NodeVisitor

Create a visitor class to systematically traverse the tree:

```python
import ast

class BinaryOpFinder(ast.NodeVisitor):
    """Find all binary operations in the code."""

    def __init__(self):
        self.operations = []

    def visit_BinOp(self, node):
        op_name = node.op.__class__.__name__
        self.operations.append({
            'operator': op_name,
            'line': node.lineno,
            'col': node.col_offset
        })
        # Continue visiting child nodes
        self.generic_visit(node)

# Usage
source = """
x = a + b
y = c * d
z = (x + y) / 2
"""

tree = ast.parse(source)
finder = BinaryOpFinder()
finder.visit(tree)

for op in finder.operations:
    print(f"Line {op['line']}: {op['operator']}")
```

**Output:**
```
Line 2: Add
Line 3: Mult
Line 4: Add
Line 4: Div
```

### Finding Specific Patterns

#### Find All Comparisons

```python
class ComparisonFinder(ast.NodeVisitor):
    """Find all comparison operations."""

    def __init__(self):
        self.comparisons = []

    def visit_Compare(self, node):
        for op in node.ops:
            self.comparisons.append({
                'operator': op.__class__.__name__,
                'line': node.lineno,
                'node': node
            })
        self.generic_visit(node)
```

#### Find All Function Calls

```python
class FunctionCallFinder(ast.NodeVisitor):
    """Find all function calls."""

    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = "<unknown>"

        self.calls.append({
            'name': func_name,
            'line': node.lineno,
            'num_args': len(node.args)
        })
        self.generic_visit(node)
```

#### Find Return Statements

```python
class ReturnFinder(ast.NodeVisitor):
    """Find all return statements."""

    def __init__(self):
        self.returns = []

    def visit_Return(self, node):
        self.returns.append({
            'line': node.lineno,
            'has_value': node.value is not None,
            'node': node
        })
        self.generic_visit(node)
```

### Pattern Matching with Context

```python
class ConditionalReturnFinder(ast.NodeVisitor):
    """Find return statements inside if blocks."""

    def __init__(self):
        self.conditional_returns = []
        self._in_if = False

    def visit_If(self, node):
        old_in_if = self._in_if
        self._in_if = True
        self.generic_visit(node)
        self._in_if = old_in_if

    def visit_Return(self, node):
        if self._in_if:
            self.conditional_returns.append({
                'line': node.lineno,
                'node': node
            })
        self.generic_visit(node)
```

---

## Replacing Code (Mutations)

### Using ast.NodeTransformer

The `NodeTransformer` class allows you to modify the AST:

```python
import ast

class ArithmeticMutator(ast.NodeTransformer):
    """Replace + with - operators."""

    def visit_BinOp(self, node):
        # First, visit children
        self.generic_visit(node)

        # Replace Add with Sub
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()

        return node

# Usage
source = "result = a + b"
tree = ast.parse(source)

mutator = ArithmeticMutator()
mutated_tree = mutator.visit(tree)

# Fix missing line numbers
ast.fix_missing_locations(mutated_tree)

print(ast.unparse(mutated_tree))
# Output: result = a - b
```

### Comprehensive Operator Mutator

```python
import ast
import copy

class OperatorMutator(ast.NodeTransformer):
    """Mutate operators for mutation testing."""

    # Define mutation mappings
    BINARY_OP_MUTATIONS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Div, ast.Add],
        ast.Div: [ast.Mult, ast.FloorDiv],
        ast.FloorDiv: [ast.Div],
        ast.Mod: [ast.Div],
    }

    COMPARE_OP_MUTATIONS = {
        ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
        ast.GtE: [ast.Gt, ast.LtE],
        ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
        ast.LtE: [ast.Lt, ast.GtE],
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq],
    }

    def __init__(self, mutation_type='binary', target_line=None):
        self.mutation_type = mutation_type
        self.target_line = target_line
        self.mutations_applied = []

    def visit_BinOp(self, node):
        self.generic_visit(node)

        if self.mutation_type != 'binary':
            return node

        if self.target_line and node.lineno != self.target_line:
            return node

        op_type = type(node.op)
        if op_type in self.BINARY_OP_MUTATIONS:
            # Get first mutation option
            new_op_type = self.BINARY_OP_MUTATIONS[op_type][0]
            old_op = node.op.__class__.__name__
            node.op = new_op_type()
            new_op = node.op.__class__.__name__
            self.mutations_applied.append({
                'line': node.lineno,
                'from': old_op,
                'to': new_op
            })

        return node

    def visit_Compare(self, node):
        self.generic_visit(node)

        if self.mutation_type != 'compare':
            return node

        if self.target_line and node.lineno != self.target_line:
            return node

        new_ops = []
        for op in node.ops:
            op_type = type(op)
            if op_type in self.COMPARE_OP_MUTATIONS:
                new_op_type = self.COMPARE_OP_MUTATIONS[op_type][0]
                old_op = op.__class__.__name__
                new_op = new_op_type()
                new_ops.append(new_op)
                self.mutations_applied.append({
                    'line': node.lineno,
                    'from': old_op,
                    'to': new_op.__class__.__name__
                })
            else:
                new_ops.append(op)

        node.ops = new_ops
        return node


# Example usage
source = """
def check_value(x, y):
    total = x + y
    if total > 10:
        return True
    return False
"""

# Mutate binary operators
tree = ast.parse(source)
mutator = OperatorMutator(mutation_type='binary')
mutated = mutator.visit(tree)
ast.fix_missing_locations(mutated)

print("Binary mutation:")
print(ast.unparse(mutated))
print(f"Applied: {mutator.mutations_applied}")

# Mutate comparison operators
tree = ast.parse(source)
mutator = OperatorMutator(mutation_type='compare')
mutated = mutator.visit(tree)
ast.fix_missing_locations(mutated)

print("\nComparison mutation:")
print(ast.unparse(mutated))
print(f"Applied: {mutator.mutations_applied}")
```

### Mutating Boolean Operators

```python
class BooleanMutator(ast.NodeTransformer):
    """Mutate boolean operators (and/or)."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.And):
            node.op = ast.Or()
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()

        return node

# Example
source = "result = a > 0 and b > 0"
tree = ast.parse(source)
mutated = BooleanMutator().visit(tree)
ast.fix_missing_locations(mutated)
print(ast.unparse(mutated))
# Output: result = a > 0 or b > 0
```

### Mutating Constants

```python
class ConstantMutator(ast.NodeTransformer):
    """Mutate constant values."""

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            # Flip booleans
            node.value = not node.value
        elif isinstance(node.value, int):
            # Mutate integers
            if node.value == 0:
                node.value = 1
            elif node.value == 1:
                node.value = 0
            else:
                node.value = node.value + 1
        elif isinstance(node.value, str):
            # Empty strings or mutate
            if node.value == "":
                node.value = "mutated"
            else:
                node.value = ""

        return node

# Example
source = """
x = 0
y = True
z = "hello"
"""
tree = ast.parse(source)
mutated = ConstantMutator().visit(tree)
ast.fix_missing_locations(mutated)
print(ast.unparse(mutated))
```

### Mutating Return Statements

```python
class ReturnMutator(ast.NodeTransformer):
    """Mutate return statements."""

    def __init__(self, mutation='none'):
        self.mutation = mutation  # 'none', 'negate', 'empty'

    def visit_Return(self, node):
        if node.value is None:
            return node

        if self.mutation == 'none':
            # Return None instead of value
            return ast.Return(value=ast.Constant(value=None))

        elif self.mutation == 'negate':
            # Negate the return value
            return ast.Return(
                value=ast.UnaryOp(
                    op=ast.Not(),
                    operand=node.value
                )
            )

        elif self.mutation == 'empty':
            # Return empty collection based on context
            return ast.Return(
                value=ast.List(elts=[], ctx=ast.Load())
            )

        return node

# Example
source = """
def get_value():
    return 42
"""
tree = ast.parse(source)
mutated = ReturnMutator(mutation='none').visit(tree)
ast.fix_missing_locations(mutated)
print(ast.unparse(mutated))
# Output: def get_value():
#             return None
```

---

## Converting AST Back to Code

### Using ast.unparse() (Python 3.9+)

```python
import ast

source = """
def greet(name):
    return f"Hello, {name}!"
"""

tree = ast.parse(source)
# Modify the tree...

# Convert back to source code
new_source = ast.unparse(tree)
print(new_source)
```

### Manual Code Generation (Pre-3.9)

For older Python versions or more control:

```python
import ast

class CodeGenerator(ast.NodeVisitor):
    """Generate code from AST (simplified)."""

    def __init__(self):
        self.code = []
        self.indent = 0

    def _indent(self):
        return "    " * self.indent

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)
        return "\n".join(self.code)

    def visit_FunctionDef(self, node):
        args = ", ".join(arg.arg for arg in node.args.args)
        self.code.append(f"{self._indent()}def {node.name}({args}):")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1

    def visit_Return(self, node):
        if node.value:
            value = self.visit(node.value)
            self.code.append(f"{self._indent()}return {value}")
        else:
            self.code.append(f"{self._indent()}return")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = {
            ast.Add: '+', ast.Sub: '-',
            ast.Mult: '*', ast.Div: '/'
        }.get(type(node.op), '?')
        return f"{left} {op} {right}"

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        return repr(node.value)
```

### Preserving Source Formatting

For preserving original formatting, use the `astor` or `libcst` libraries:

```python
# Using astor (pip install astor)
import astor

source = """
def add(a, b):
    # Add two numbers
    return a + b
"""

tree = ast.parse(source)
# Modify tree...

# Convert back with better formatting
new_source = astor.to_source(tree)
```

---

## Practical Examples

### Example 1: Find All Mutation Points

```python
import ast
from dataclasses import dataclass
from typing import List

@dataclass
class MutationPoint:
    """Represents a point in code that can be mutated."""
    line: int
    col: int
    node_type: str
    original: str
    mutations: List[str]

class MutationPointFinder(ast.NodeVisitor):
    """Find all points in code that can be mutated."""

    BINARY_MUTATIONS = {
        'Add': ['Sub', 'Mult'],
        'Sub': ['Add', 'Mult'],
        'Mult': ['Div', 'Add'],
        'Div': ['Mult', 'FloorDiv'],
    }

    COMPARE_MUTATIONS = {
        'Gt': ['GtE', 'Lt'],
        'Lt': ['LtE', 'Gt'],
        'Eq': ['NotEq'],
        'NotEq': ['Eq'],
        'GtE': ['Gt', 'Lt'],
        'LtE': ['Lt', 'Gt'],
    }

    def __init__(self):
        self.mutation_points: List[MutationPoint] = []

    def visit_BinOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name in self.BINARY_MUTATIONS:
            self.mutation_points.append(MutationPoint(
                line=node.lineno,
                col=node.col_offset,
                node_type='BinOp',
                original=op_name,
                mutations=self.BINARY_MUTATIONS[op_name]
            ))
        self.generic_visit(node)

    def visit_Compare(self, node):
        for i, op in enumerate(node.ops):
            op_name = op.__class__.__name__
            if op_name in self.COMPARE_MUTATIONS:
                self.mutation_points.append(MutationPoint(
                    line=node.lineno,
                    col=node.col_offset,
                    node_type='Compare',
                    original=op_name,
                    mutations=self.COMPARE_MUTATIONS[op_name]
                ))
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        op_name = node.op.__class__.__name__
        mutation = 'Or' if op_name == 'And' else 'And'
        self.mutation_points.append(MutationPoint(
            line=node.lineno,
            col=node.col_offset,
            node_type='BoolOp',
            original=op_name,
            mutations=[mutation]
        ))
        self.generic_visit(node)


# Usage
source = """
def validate(x, y):
    if x > 0 and y > 0:
        result = x + y
        if result >= 10:
            return True
    return False
"""

tree = ast.parse(source)
finder = MutationPointFinder()
finder.visit(tree)

print("Mutation Points Found:")
print("-" * 60)
for point in finder.mutation_points:
    print(f"Line {point.line}: {point.node_type}")
    print(f"  Original: {point.original}")
    print(f"  Can mutate to: {point.mutations}")
    print()
```

### Example 2: Targeted Mutation at Specific Location

```python
import ast
import copy

class TargetedMutator(ast.NodeTransformer):
    """Apply a specific mutation at a specific location."""

    def __init__(self, target_line, target_col, from_op, to_op):
        self.target_line = target_line
        self.target_col = target_col
        self.from_op = from_op
        self.to_op = to_op
        self.mutation_applied = False

    def _get_op_class(self, op_name):
        """Get AST operator class from name."""
        ops = {
            'Add': ast.Add, 'Sub': ast.Sub,
            'Mult': ast.Mult, 'Div': ast.Div,
            'Gt': ast.Gt, 'Lt': ast.Lt,
            'GtE': ast.GtE, 'LtE': ast.LtE,
            'Eq': ast.Eq, 'NotEq': ast.NotEq,
            'And': ast.And, 'Or': ast.Or,
        }
        return ops.get(op_name)

    def visit_BinOp(self, node):
        self.generic_visit(node)

        if (node.lineno == self.target_line and
            node.col_offset == self.target_col and
            node.op.__class__.__name__ == self.from_op):

            new_op_class = self._get_op_class(self.to_op)
            if new_op_class:
                node.op = new_op_class()
                self.mutation_applied = True

        return node

    def visit_Compare(self, node):
        self.generic_visit(node)

        if (node.lineno == self.target_line and
            node.col_offset == self.target_col):

            new_ops = []
            for op in node.ops:
                if op.__class__.__name__ == self.from_op:
                    new_op_class = self._get_op_class(self.to_op)
                    if new_op_class:
                        new_ops.append(new_op_class())
                        self.mutation_applied = True
                    else:
                        new_ops.append(op)
                else:
                    new_ops.append(op)
            node.ops = new_ops

        return node


def apply_mutation(source, line, col, from_op, to_op):
    """Apply a specific mutation to source code."""
    tree = ast.parse(source)
    mutator = TargetedMutator(line, col, from_op, to_op)
    mutated_tree = mutator.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(mutated_tree)

    if mutator.mutation_applied:
        return ast.unparse(mutated_tree)
    return None


# Usage
source = """
def calculate(a, b):
    total = a + b
    if total > 10:
        return total * 2
    return total
"""

# Mutate the + on line 3 to -
mutated = apply_mutation(source, 3, 12, 'Add', 'Sub')
if mutated:
    print("Mutated code:")
    print(mutated)
```

### Example 3: Generate All Possible Mutations

```python
import ast
import copy
from typing import Generator, Tuple

def generate_all_mutations(source: str) -> Generator[Tuple[str, str, int], None, None]:
    """
    Generate all possible mutations for the given source code.

    Yields: (mutated_source, description, line_number)
    """
    tree = ast.parse(source)

    # Find all mutation points
    finder = MutationPointFinder()
    finder.visit(tree)

    for point in finder.mutation_points:
        for mutation in point.mutations:
            mutated = apply_mutation(
                source,
                point.line,
                point.col,
                point.original,
                mutation
            )
            if mutated:
                desc = f"Line {point.line}: {point.original} -> {mutation}"
                yield (mutated, desc, point.line)


# Usage
source = """
def is_valid(x, y):
    if x > 0 and y > 0:
        return x + y
    return 0
"""

print("All possible mutations:")
print("=" * 60)
for i, (mutated_code, description, line) in enumerate(generate_all_mutations(source), 1):
    print(f"\nMutation {i}: {description}")
    print("-" * 40)
    print(mutated_code)
```

---

## Complete Mutation Testing Example

Here's a complete, runnable mutation testing implementation:

```python
#!/usr/bin/env python3
"""
Simple Mutation Testing Framework using AST

This module demonstrates how to:
1. Parse Python code into AST
2. Find mutation points
3. Apply mutations
4. Run tests against mutants
"""

import ast
import copy
import sys
import importlib.util
from dataclasses import dataclass
from typing import List, Callable, Any
from pathlib import Path
import tempfile
import traceback


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MutationPoint:
    """A location in code that can be mutated."""
    line: int
    col: int
    end_line: int
    end_col: int
    node_type: str
    original_op: str
    possible_mutations: List[str]


@dataclass
class Mutation:
    """A specific mutation to apply."""
    point: MutationPoint
    new_op: str

    @property
    def description(self) -> str:
        return f"Line {self.point.line}: {self.point.original_op} -> {self.new_op}"


@dataclass
class MutationResult:
    """Result of testing a mutation."""
    mutation: Mutation
    killed: bool
    error_message: str = ""


# =============================================================================
# AST Visitors
# =============================================================================

class MutationPointFinder(ast.NodeVisitor):
    """Find all points in code that can be mutated."""

    BINARY_OP_MUTATIONS = {
        'Add': ['Sub', 'Mult'],
        'Sub': ['Add'],
        'Mult': ['Div', 'FloorDiv'],
        'Div': ['Mult'],
        'FloorDiv': ['Div'],
        'Mod': ['Div'],
    }

    COMPARE_MUTATIONS = {
        'Eq': ['NotEq'],
        'NotEq': ['Eq'],
        'Lt': ['LtE', 'Gt'],
        'LtE': ['Lt', 'GtE'],
        'Gt': ['GtE', 'Lt'],
        'GtE': ['Gt', 'LtE'],
    }

    BOOL_OP_MUTATIONS = {
        'And': ['Or'],
        'Or': ['And'],
    }

    def __init__(self):
        self.points: List[MutationPoint] = []

    def visit_BinOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name in self.BINARY_OP_MUTATIONS:
            self.points.append(MutationPoint(
                line=node.lineno,
                col=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                end_col=node.end_col_offset or node.col_offset,
                node_type='BinOp',
                original_op=op_name,
                possible_mutations=self.BINARY_OP_MUTATIONS[op_name]
            ))
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            op_name = op.__class__.__name__
            if op_name in self.COMPARE_MUTATIONS:
                self.points.append(MutationPoint(
                    line=node.lineno,
                    col=node.col_offset,
                    end_line=node.end_lineno or node.lineno,
                    end_col=node.end_col_offset or node.col_offset,
                    node_type='Compare',
                    original_op=op_name,
                    possible_mutations=self.COMPARE_MUTATIONS[op_name]
                ))
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name in self.BOOL_OP_MUTATIONS:
            self.points.append(MutationPoint(
                line=node.lineno,
                col=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                end_col=node.end_col_offset or node.col_offset,
                node_type='BoolOp',
                original_op=op_name,
                possible_mutations=self.BOOL_OP_MUTATIONS[op_name]
            ))
        self.generic_visit(node)


class MutationApplier(ast.NodeTransformer):
    """Apply a specific mutation to the AST."""

    OP_CLASSES = {
        # Binary operators
        'Add': ast.Add, 'Sub': ast.Sub,
        'Mult': ast.Mult, 'Div': ast.Div,
        'FloorDiv': ast.FloorDiv, 'Mod': ast.Mod,
        # Comparison operators
        'Eq': ast.Eq, 'NotEq': ast.NotEq,
        'Lt': ast.Lt, 'LtE': ast.LtE,
        'Gt': ast.Gt, 'GtE': ast.GtE,
        # Boolean operators
        'And': ast.And, 'Or': ast.Or,
    }

    def __init__(self, mutation: Mutation):
        self.mutation = mutation
        self.applied = False

    def _matches_location(self, node) -> bool:
        return (node.lineno == self.mutation.point.line and
                node.col_offset == self.mutation.point.col)

    def visit_BinOp(self, node):
        self.generic_visit(node)

        if (self.mutation.point.node_type == 'BinOp' and
            self._matches_location(node) and
            node.op.__class__.__name__ == self.mutation.point.original_op):

            new_op_class = self.OP_CLASSES.get(self.mutation.new_op)
            if new_op_class:
                node.op = new_op_class()
                self.applied = True

        return node

    def visit_Compare(self, node):
        self.generic_visit(node)

        if (self.mutation.point.node_type == 'Compare' and
            self._matches_location(node)):

            new_ops = []
            for op in node.ops:
                if op.__class__.__name__ == self.mutation.point.original_op:
                    new_op_class = self.OP_CLASSES.get(self.mutation.new_op)
                    if new_op_class:
                        new_ops.append(new_op_class())
                        self.applied = True
                    else:
                        new_ops.append(op)
                else:
                    new_ops.append(op)
            node.ops = new_ops

        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        if (self.mutation.point.node_type == 'BoolOp' and
            self._matches_location(node) and
            node.op.__class__.__name__ == self.mutation.point.original_op):

            new_op_class = self.OP_CLASSES.get(self.mutation.new_op)
            if new_op_class:
                node.op = new_op_class()
                self.applied = True

        return node


# =============================================================================
# Core Functions
# =============================================================================

def find_mutation_points(source: str) -> List[MutationPoint]:
    """Find all mutation points in source code."""
    tree = ast.parse(source)
    finder = MutationPointFinder()
    finder.visit(tree)
    return finder.points


def generate_mutations(source: str) -> List[Mutation]:
    """Generate all possible mutations for source code."""
    points = find_mutation_points(source)
    mutations = []

    for point in points:
        for new_op in point.possible_mutations:
            mutations.append(Mutation(point=point, new_op=new_op))

    return mutations


def apply_mutation(source: str, mutation: Mutation) -> str:
    """Apply a mutation to source code and return mutated source."""
    tree = ast.parse(source)
    applier = MutationApplier(mutation)
    mutated_tree = applier.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(mutated_tree)

    if applier.applied:
        return ast.unparse(mutated_tree)
    return None


def load_module_from_source(source: str, module_name: str):
    """Load a Python module from source code string."""
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)

    exec(source, module.__dict__)
    return module


def run_tests(source: str, test_func: Callable) -> bool:
    """Run tests against the given source code.

    Returns True if tests pass, False otherwise.
    """
    try:
        module = load_module_from_source(source, "test_module")
        test_func(module)
        return True
    except AssertionError:
        return False
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        return False


def run_mutation_testing(source: str, test_func: Callable) -> dict:
    """Run complete mutation testing suite.

    Args:
        source: The source code to test
        test_func: A callable that takes a module and runs tests on it

    Returns:
        Dictionary with mutation testing results
    """
    mutations = generate_mutations(source)
    results = []

    print(f"Found {len(mutations)} mutations...")
    print()

    for i, mutation in enumerate(mutations, 1):
        mutated_source = apply_mutation(source, mutation)

        if mutated_source is None:
            continue

        killed = run_tests(mutated_source, test_func)
        result = MutationResult(mutation=mutation, killed=killed)
        results.append(result)

        status = "KILLED" if killed else "SURVIVED"
        print(f"[{i}/{len(mutations)}] {mutation.description} ... {status}")

    # Calculate scores
    killed_count = sum(1 for r in results if r.killed)
    mutation_score = killed_count / len(results) if results else 0

    return {
        'total_mutations': len(results),
        'killed': killed_count,
        'survived': len(results) - killed_count,
        'mutation_score': mutation_score,
        'results': results
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example source code to test
    source = """
def add(a, b):
    return a + b

def greater_than(x, y):
    return x > y

def validate(x):
    if x > 0 and x < 100:
        return True
    return False
"""

    # Example test function
    def test_suite(module):
        # Test add function
        assert module.add(2, 3) == 5
        assert module.add(-1, 1) == 0

        # Test greater_than function
        assert module.greater_than(5, 3) == True
        assert module.greater_than(2, 5) == False

        # Test validate function
        assert module.validate(50) == True
        assert module.validate(150) == False
        assert module.validate(0) == False

    # Run mutation testing
    print("=" * 60)
    print("MUTATION TESTING REPORT")
    print("=" * 60)
    print()

    results = run_mutation_testing(source, test_suite)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total mutations: {results['total_mutations']}")
    print(f"Killed: {results['killed']}")
    print(f"Survived: {results['survived']}")
    print(f"Mutation Score: {results['mutation_score']:.2%}")
    print("=" * 60)
```

This complete example includes:
- Finding all mutation points
- Applying mutations
- Running test suites against mutations
- Calculating mutation scores

Use this framework as a foundation for building your own mutation testing tools!