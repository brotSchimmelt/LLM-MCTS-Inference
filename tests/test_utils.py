import pytest

from llm_mcts_inference.utils.utils import is_numeric_score, normalize_rating_score


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
    assert normalize_rating_score(input_score) == pytest.approx(
        expected_output, rel=1e-6
    )


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
