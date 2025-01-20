import pytest

from llm_mcts_inference.utils.utils import (
    extract_first_number,
    is_numeric_score,
    normalize_rating_score,
)


@pytest.mark.parametrize(
    "input_score, expected_output",
    [
        (42, 0.42),
        (95, 0.95),
        (0, 0.0),
        (-10, 0.0),
        (100, 0.95),
        ("85", 0.85),
        ("0", 0.0),
        ("-20", 0.0),
        ("120", 0.95),
    ],
)
def test_normalize_rating_score(input_score, expected_output):
    """
    Test the normalize_rating_score function with various inputs.
    """
    assert normalize_rating_score(input_score) == pytest.approx(expected_output, rel=1e-6)


def test_invalid_inputs():
    """
    Test that invalid inputs raise appropriate exceptions.
    """
    with pytest.raises(ValueError):
        normalize_rating_score("invalid")

    with pytest.raises(TypeError):
        normalize_rating_score(None)

    with pytest.raises(ValueError):
        normalize_rating_score("")


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("123", True),
        ("123.45", True),
        ("0", True),
        (".123", True),
        ("123.", True),
        ("123.45.67", False),
        ("abc", False),
        ("", False),
        ("123abc", False),
        ("-123", True),
        ("1.23e10", False),
        (" ", False),
        (".", False),
    ],
)
def test_is_numeric_score(input_string, expected_output):
    """
    Test the is_numeric_score function with various inputs.
    """
    assert is_numeric_score(input_string) == expected_output


def test_extract_first_number_basic():
    """
    Test extracting the first number from a standard string with multiple numbers.
    """
    assert extract_first_number("this is a test with 100 and 42") == 100


def test_extract_first_number_no_numbers():
    """
    Test extracting when the string contains no numbers.
    """
    assert extract_first_number("no numbers here") == 0


def test_extract_first_number_at_start():
    """
    Test extracting the first number when it appears at the start of the string.
    """
    assert extract_first_number("42 is the answer") == 42


def test_extract_first_number_with_symbols():
    """
    Test extracting when the string contains symbols and numbers.
    """
    assert extract_first_number("price: $50, discount: 20%") == 50


def test_extract_first_number_empty_string():
    """
    Test handling an empty string.
    """
    assert extract_first_number("") == 0
