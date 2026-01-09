"""Simple calculator module for mutation testing demo."""


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
