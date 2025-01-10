from typing import Union


def normalize_rating_score(score: Union[int, str]) -> float:
    """
    Normalize a rating score to a float value between 0 and 0.95.

    Args:
        score (int | str): The rating score to normalize.
            Can be an integer, or string representation of a number.

    Returns:
        float: The normalized score, capped between 0 and 0.95.
    """
    if score is None:
        raise TypeError("Input score cannot be None.")

    if not isinstance(score, (int, str)):
        raise ValueError("Input score must be an integer or string.")

    if isinstance(score, str):
        if not is_numeric_score(score):
            raise ValueError(f"Input score must be a numeric string. Found: {score}")

    capped_score = max(0, min(95, float(score)))
    return capped_score / 100.0


def is_numeric_score(s: str) -> bool:
    """
    Check if a string represents a numeric value, including integers or decimals.

    The function allows:
    - Positive and negative numbers.
    - A single decimal point.

    Args:
        s (str): The input string to check.

    Returns:
        bool: True if the string represents a valid number (integer or decimal),
              False otherwise.
    """
    s = s.replace("-", "", 1)
    return s.replace(".", "", 1).isdigit()
