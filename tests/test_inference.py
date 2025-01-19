from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llm_mcts_inference.inference import (
    _validate_request_attributes,
    get_model_response,
    get_structured_model_response,
)


def test_validate_request_attributes_basic():
    """
    Test _validate_request_attributes with a standard model name.
    """
    prompt = "What is the weather today?"
    model_settings = {"model_name": "gpt-3.5-turbo"}

    expected_result = {
        "model": "gpt-3.5-turbo",
        "messages": [{"content": prompt, "role": "user"}],
    }

    result = _validate_request_attributes(prompt, model_settings)
    assert result == expected_result, "Basic model configuration failed."


def test_validate_request_attributes_with_ollama_model():
    """
    Test _validate_request_attributes when model name contains 'ollama'.
    """
    prompt = "Summarize this text."
    model_settings = {"model_name": "ollama-custom-model"}

    expected_result = {
        "model": "ollama-custom-model",
        "messages": [{"content": prompt, "role": "user"}],
        "api_base": "http://localhost:11434",
    }

    result = _validate_request_attributes(prompt, model_settings)
    assert result == expected_result, "'ollama' model configuration failed."


def test_validate_request_attributes_missing_model_name():
    """
    Test _validate_request_attributes when 'model_name' is missing from model_settings.
    """
    prompt = "Who won the match yesterday?"
    model_settings = {}

    with pytest.raises(KeyError) as exc_info:
        _validate_request_attributes(prompt, model_settings)

    assert "model_name" in str(exc_info.value), "Missing 'model_name' key did not raise KeyError."


def test_validate_request_attributes_empty_prompt():
    """
    Test _validate_request_attributes with an empty prompt.
    """
    prompt = ""
    model_settings = {"model_name": "gpt-3.5-turbo"}

    expected_result = {
        "model": "gpt-3.5-turbo",
        "messages": [{"content": prompt, "role": "user"}],
    }

    result = _validate_request_attributes(prompt, model_settings)
    assert result == expected_result, "Empty prompt configuration failed."


def test_validate_request_attributes_invalid_model_name():
    """
    Test _validate_request_attributes with a non-standard model name.
    """
    prompt = "Tell me a joke."
    model_settings = {"model_name": "nonexistent-model"}

    expected_result = {
        "model": "nonexistent-model",
        "messages": [{"content": prompt, "role": "user"}],
    }

    result = _validate_request_attributes(prompt, model_settings)
    assert result == expected_result, "Non-standard model configuration failed."


def test_get_model_response_basic():
    """
    Test get_model_response with a basic prompt and mocked API response.
    """
    prompt = "What is the capital of France?"
    model_settings = {"model_name": "gpt-3.5-turbo"}

    mock_response = {"choices": [{"message": {"content": "The capital of France is Paris."}}]}

    with patch(
        "llm_mcts_inference.inference.litellm.completion", return_value=mock_response
    ) as mock_completion:
        response = get_model_response(prompt, model_settings)

        expected_attributes = _validate_request_attributes(prompt, model_settings)
        mock_completion.assert_called_once_with(**expected_attributes)

        assert response == "The capital of France is Paris.", "Response content mismatch."


def test_get_model_response_empty_prompt():
    """
    Test get_model_response with an empty prompt.
    """
    prompt = ""
    model_settings = {"model_name": "gpt-3.5-turbo"}

    mock_response = {"choices": [{"message": {"content": "No input provided."}}]}

    with patch(
        "llm_mcts_inference.inference.litellm.completion", return_value=mock_response
    ) as mock_completion:
        response = get_model_response(prompt, model_settings)

        expected_attributes = _validate_request_attributes(prompt, model_settings)
        mock_completion.assert_called_once_with(**expected_attributes)

        assert response == "No input provided.", "Response content mismatch for empty prompt."


def test_get_model_response_with_ollama_model():
    """
    Test get_model_response when using an Ollama model.
    """
    prompt = "Summarize the document."
    model_settings = {"model_name": "ollama-custom-model"}

    mock_response = {
        "choices": [{"message": {"content": "The document discusses AI advancements."}}]
    }

    with patch(
        "llm_mcts_inference.inference.litellm.completion", return_value=mock_response
    ) as mock_completion:
        response = get_model_response(prompt, model_settings)

        expected_attributes = _validate_request_attributes(prompt, model_settings)
        mock_completion.assert_called_once_with(**expected_attributes)

        assert (
            response == "The document discusses AI advancements."
        ), "Response content mismatch for Ollama model."


def test_get_model_response_invalid_response_structure():
    """
    Test get_model_response with an invalid API response structure.
    """
    prompt = "What is the square root of 16?"
    model_settings = {"model_name": "gpt-3.5-turbo"}

    mock_response = {}

    with patch(
        "llm_mcts_inference.inference.litellm.completion", return_value=mock_response
    ) as mock_completion:
        with pytest.raises(KeyError):
            get_model_response(prompt, model_settings)

        expected_attributes = _validate_request_attributes(prompt, model_settings)
        mock_completion.assert_called_once_with(**expected_attributes)


def test_get_structured_model_response():
    """
    Test get_structured_model_response with a mocked API response.
    """

    class User(BaseModel):
        name: str
        age: int

    prompt = "Extract the user details: Jason is 25 years old."
    model_settings = {"model_name": "gpt-3.5-turbo"}

    mock_response = User(name="Jason", age=25)

    with patch("llm_mcts_inference.inference.litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        with patch("llm_mcts_inference.inference.instructor.from_litellm") as mock_instructor:
            mock_client = mock_instructor.return_value
            mock_client.create.return_value = mock_response

            response = get_structured_model_response(prompt, model_settings, User)

            assert response == mock_response, "Structured response mismatch"

            expected_attributes = _validate_request_attributes(prompt, model_settings)
            expected_attributes["response_model"] = User
            mock_client.create.assert_called_once_with(**expected_attributes)


def test_get_structured_model_response_invalid_schema():
    """
    Test get_structured_model_response with an invalid schema in the API response.
    """

    class User(BaseModel):
        name: str
        age: int

    prompt = "Extract the user details: Jason is 25 years old."
    model_settings = {"model_name": "gpt-3.5-turbo"}

    invalid_response = {"name": "Jason"}

    with patch("llm_mcts_inference.inference.litellm.completion") as mock_completion:
        mock_completion.return_value = invalid_response

        with patch("llm_mcts_inference.inference.instructor.from_litellm") as mock_instructor:
            mock_client = mock_instructor.return_value
            mock_client.create.side_effect = ValueError("Validation failed for response schema.")

            with pytest.raises(ValueError, match="Validation failed for response schema."):
                get_structured_model_response(prompt, model_settings, User)
