import pickle
from typing import Any

import main
import numpy as np
import pytest
import scipy as sp

try:
    with open("expected", "rb") as f:
        expected = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: The 'expected' file was not found. Please ensure it is in the correct directory."
    )
    expected = {"is_diagonally_dominant": [], "residual_norm": []}


# --- Data Preparation ---

valid_is_diagonally_dominant = [
    (A, res) for A, res in expected["is_diagonally_dominant"] if res is not None
]
invalid_is_diagonally_dominant = [
    (A, res) for A, res in expected["is_diagonally_dominant"] if res is None
]

valid_residual_norm = [
    (A, x, b, res) for A, x, b, res in expected["residual_norm"] if res is not None
]
invalid_residual_norm = [
    (A, x, b, res) for A, x, b, res in expected["residual_norm"] if res is None
]


# --- Tests for is_diagonally_dominant ---


@pytest.mark.parametrize("A, expected_result", invalid_is_diagonally_dominant)
def test_is_diagonally_dominant_invalid_input(
    A: Any, expected_result: None
):
    """Tests if is_diagonally_dominant correctly handles invalid input data by returning None."""
    actual = main.is_diagonally_dominant(A)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("A, expected_result", valid_is_diagonally_dominant)
def test_is_diagonally_dominant_correct_solution(
    A: np.ndarray | sp.sparse.csc_array, expected_result: bool
):
    """Tests if is_diagonally_dominant correctly identifies diagonally dominant matrices."""
    actual_result = main.is_diagonally_dominant(A)
    assert actual_result == expected_result, (
        f"Expected {expected_result} for the given matrix, but got {actual_result}."
    )


# --- Tests for residual_norm ---


@pytest.mark.parametrize("A, x, b, expected_result", invalid_residual_norm)
def test_residual_norm_invalid_input(
    A: Any, x: Any, b: Any, expected_result: None
):
    """Tests if residual_norm correctly handles invalid input data by returning None."""
    actual = main.residual_norm(A, x, b)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("A, x, b, expected_result", valid_residual_norm)
def test_residual_norm_correct_solution(
    A: np.ndarray, x: np.ndarray, b: np.ndarray, expected_result: float
):
    """Tests if residual_norm calculates the correct residual norm for valid inputs."""
    actual_result = main.residual_norm(A, x, b)
    assert actual_result == pytest.approx(expected_result), (
        f"Expected norm {expected_result}, but got {actual_result}."
    )