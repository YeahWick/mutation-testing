"""Tests for calculator module."""

import pytest
from src.calculator import add, subtract, is_positive, clamp


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0


def test_is_positive():
    assert is_positive(5) is True
    assert is_positive(-1) is False
    # Note: Missing test for 0 - mutation will survive!


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
