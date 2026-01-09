"""Core mutation testing functionality using AST transformation."""

import ast
import copy
import sys
import types
from dataclasses import dataclass
from typing import Callable, List, Optional, Any


@dataclass
class Mutation:
    """A mutation to apply to code."""
    id: str
    function: str
    original: str
    mutant: str
    description: str = ""
    line: Optional[int] = None


@dataclass
class MutationResult:
    """Result of testing a mutation."""
    mutation: Mutation
    killed: bool
    error: Optional[str] = None


class PatternReplacer(ast.NodeTransformer):
    """Replace AST patterns in code."""

    def __init__(self, original_ast: ast.AST, replacement_ast: ast.AST):
        self.original = original_ast
        self.replacement = replacement_ast
        self.replaced = False

    def visit(self, node):
        if self._matches(node, self.original):
            self.replaced = True
            return ast.copy_location(copy.deepcopy(self.replacement), node)
        return self.generic_visit(node)

    def _matches(self, node: ast.AST, pattern: ast.AST) -> bool:
        """Check if node matches pattern structurally."""
        if type(node) != type(pattern):
            return False

        if isinstance(pattern, ast.Name):
            return node.id == pattern.id

        if isinstance(pattern, ast.Constant):
            return node.value == pattern.value

        if isinstance(pattern, ast.BinOp):
            return (
                type(node.op) == type(pattern.op)
                and self._matches(node.left, pattern.left)
                and self._matches(node.right, pattern.right)
            )

        if isinstance(pattern, ast.Compare):
            if len(node.ops) != len(pattern.ops):
                return False
            if not all(type(a) == type(b) for a, b in zip(node.ops, pattern.ops)):
                return False
            if not self._matches(node.left, pattern.left):
                return False
            return all(
                self._matches(a, b)
                for a, b in zip(node.comparators, pattern.comparators)
            )

        if isinstance(pattern, ast.Return):
            if pattern.value is None:
                return node.value is None
            if node.value is None:
                return False
            return self._matches(node.value, pattern.value)

        if isinstance(pattern, ast.UnaryOp):
            return type(node.op) == type(pattern.op) and self._matches(
                node.operand, pattern.operand
            )

        if isinstance(pattern, ast.BoolOp):
            if type(node.op) != type(pattern.op):
                return False
            if len(node.values) != len(pattern.values):
                return False
            return all(
                self._matches(a, b) for a, b in zip(node.values, pattern.values)
            )

        # Default: try to match all child fields
        for field, value in ast.iter_fields(pattern):
            node_value = getattr(node, field, None)
            if isinstance(value, ast.AST):
                if not isinstance(node_value, ast.AST):
                    return False
                if not self._matches(node_value, value):
                    return False
            elif isinstance(value, list):
                if not isinstance(node_value, list) or len(value) != len(node_value):
                    return False
                for v, nv in zip(value, node_value):
                    if isinstance(v, ast.AST):
                        if not self._matches(nv, v):
                            return False

        return True


class MutationInjector:
    """Inject mutations into loaded modules at runtime."""

    def __init__(self):
        self._original_code: dict[str, types.CodeType] = {}

    def inject(
        self,
        module_name: str,
        function_name: str,
        original: str,
        mutant: str,
    ) -> bool:
        """
        Inject a mutation into a function.

        Args:
            module_name: The module containing the function
            function_name: Name of function to mutate
            original: Original code pattern to find
            mutant: Code to replace it with

        Returns:
            True if mutation was applied
        """
        module = sys.modules.get(module_name)
        if not module:
            return False

        func = getattr(module, function_name, None)
        if not callable(func):
            return False

        key = f"{module_name}.{function_name}"
        if key not in self._original_code:
            self._original_code[key] = func.__code__

        # Get function source
        import inspect

        try:
            source = inspect.getsource(func)
        except OSError:
            return False

        # Apply mutation
        mutated_source = self._apply_mutation(source, original, mutant)
        if not mutated_source:
            return False

        # Compile and inject
        new_code = self._compile_function(mutated_source, function_name, func)
        if not new_code:
            return False

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
        """Restore all mutated functions."""
        for key in list(self._original_code.keys()):
            module_name, function_name = key.rsplit(".", 1)
            self.restore(module_name, function_name)
        self._original_code.clear()

    def _parse_pattern(self, pattern: str) -> Optional[ast.AST]:
        """Parse a code pattern to AST."""
        try:
            tree = ast.parse(pattern, mode="eval")
            return tree.body
        except SyntaxError:
            pass

        try:
            tree = ast.parse(pattern, mode="exec")
            return tree.body[0] if tree.body else None
        except SyntaxError:
            return None

    def _apply_mutation(
        self, source: str, original: str, mutant: str
    ) -> Optional[str]:
        """Apply mutation to source code."""
        tree = ast.parse(source)

        original_ast = self._parse_pattern(original)
        mutant_ast = self._parse_pattern(mutant)

        if not original_ast or not mutant_ast:
            return None

        transformer = PatternReplacer(original_ast, mutant_ast)
        new_tree = transformer.visit(tree)

        if not transformer.replaced:
            return None

        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)

    def _compile_function(
        self, source: str, func_name: str, original_func: Callable
    ) -> Optional[types.CodeType]:
        """Compile mutated source and extract code object."""
        try:
            code = compile(source, "<mutation>", "exec")
            namespace = {"__builtins__": __builtins__}
            # Copy globals from original function
            if hasattr(original_func, "__globals__"):
                namespace.update(original_func.__globals__)
            exec(code, namespace)
            func = namespace.get(func_name)
            return func.__code__ if func else None
        except Exception:
            return None


class MutationError(Exception):
    """Raised when a mutation cannot be applied."""

    pass


def run_mutation_tests(
    mutations: List[Mutation],
    test_runner: Callable[[], bool],
    module_name: str,
) -> List[MutationResult]:
    """
    Run mutation tests.

    Args:
        mutations: List of mutations to test
        test_runner: Callable that runs tests and returns True if they pass
        module_name: Module containing functions to mutate

    Returns:
        List of MutationResult

    Raises:
        MutationError: If a mutation pattern doesn't match any code
    """
    injector = MutationInjector()
    results = []

    for mutation in mutations:
        # Apply mutation
        success = injector.inject(
            module_name=module_name,
            function_name=mutation.function,
            original=mutation.original,
            mutant=mutation.mutant,
        )

        if not success:
            raise MutationError(
                f"[{mutation.id}] Pattern not found in {module_name}.{mutation.function}: "
                f"'{mutation.original}'"
            )

        # Run tests
        try:
            tests_pass = test_runner()
            # If tests pass, mutation survived (bad)
            # If tests fail, mutation was killed (good)
            killed = not tests_pass
        except Exception as e:
            # Exception means test caught the mutation
            killed = True

        results.append(MutationResult(mutation=mutation, killed=killed))

        # Restore original
        injector.restore(module_name, mutation.function)

    return results
