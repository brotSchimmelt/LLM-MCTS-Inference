from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_mcts_inference.inference import get_model_response, get_structured_model_response


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client."""
    with patch("openai.OpenAI") as mock_client:
        mock_instance = MagicMock()
        mock_instance.completions.create.return_value = MagicMock(
            choices=[MagicMock(text="This is a mock response")]
        )
        mock_client.return_value = mock_instance
        yield mock_client


def test_get_model_response(mock_openai_client):
    """Test the get_model_response function."""
    model_settings = {
        "base_url": "http://localhost:11434/v1",
        "api_key": "mock_api_key",
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
    }
    prompt = "Write a short story about AI."

    response = get_model_response(prompt, model_settings)

    assert response == "This is a mock response"
    mock_openai_client.assert_called_once()
    mock_openai_client.return_value.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        seed=42,
    )


class MockSchema(BaseModel):
    output: str
    score: int


@pytest.fixture
def mock_outlines():
    """Fixture to mock the outlines library."""
    with (
        patch("outlines.models.openai") as mock_openai,
        patch("outlines.generate.json") as mock_generator,
    ):
        mock_openai.return_value = MagicMock()
        mock_generator.return_value = MagicMock(
            return_value=MockSchema(output="Structured mock response", score=95)
        )
        yield mock_openai, mock_generator


def test_get_structured_model_response(mock_outlines):
    """Test the get_structured_model_response function."""
    model_settings = {
        "model_name": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "mock_api_key",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    prompt = "Provide structured feedback and a rating."
    json_schema = MockSchema

    response = get_structured_model_response(prompt, model_settings, json_schema)

    assert response.output == "Structured mock response"
    assert response.score == 95

    mock_outlines[0].assert_called_once_with(
        "llama3", base_url="http://localhost:11434/v1", api_key="mock_api_key"
    )
    mock_outlines[1].assert_called_once()
