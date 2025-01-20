from unittest.mock import patch

from pydantic import BaseModel

from llm_mcts_inference.inference import (
    generate_feedback,
    generate_improved_version,
    generate_initial_answer,
    generate_rating,
    get_model_response,
    get_structured_model_response,
)


class MockRatingResponse(BaseModel):
    rating: float


class MockImprovedResponse(BaseModel):
    ImprovedText: str


def test_generate_initial_answer():
    """
    Test generate_initial_answer with mocked response.
    """
    prompt = "What is the capital of France?"
    request_settings = {"model_name": "gpt-3.5-turbo", "temperature": 0.7, "top_p": 0.9}

    expected_settings = {"model_name": "gpt-3.5-turbo", "temperature": 0.0, "top_p": 1.0}

    mock_response = {"choices": [{"message": {"content": "The capital of France is Paris."}}]}

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        response = generate_initial_answer(prompt, request_settings)

        assert response == "The capital of France is Paris.", "Response content mismatch."

        mock_completion.assert_called_once_with(
            messages=[{"content": prompt, "role": "user"}], **expected_settings
        )


def test_generate_rating():
    """
    Test generate_rating with mocked structured response.
    """
    prompt = "Rate the answer."
    answer = "The capital of France is Paris."
    request_settings = {"model_name": "gpt-3.5-turbo"}

    mock_rating_response = MockRatingResponse(rating=85)

    with patch(
        "llm_mcts_inference.inference.get_structured_model_response",
        return_value=mock_rating_response,
    ):
        response = generate_rating(prompt, answer, request_settings, MockRatingResponse)

        assert response == 0.85, "Normalized rating score mismatch."


def test_generate_feedback():
    """
    Test generate_feedback with mocked response.
    """
    prompt = "What is the capital of France?"
    answer = "Paris"
    request_settings = {"model_name": "gpt-3.5-turbo"}

    expected_message = (
        "\nYou are an expert assistant analyzing the user's original prompt and the provided initial answer.\n"
        "Your goal is to give clear, constructive, and concise feedback that will guide improvement.\n\n"
        "Original Prompt:\nWhat is the capital of France?\n\n"
        "Initial Answer:\nParis\n\n"
        "Instructions:\n"
        "- Provide a high-quality critique that focuses on how this answer could be improved.\n"
        "- Be concise and to the point.\n"
        "- Highlight key areas that need correction, clarification, or further detail.\n"
        "- Do not rewrite or provide the full answer; focus only on providing feedback.\n"
    )

    mock_response = {
        "choices": [{"message": {"content": "The answer is correct, but add more details."}}]
    }

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        response = generate_feedback(prompt, answer, request_settings)

        assert response == "The answer is correct, but add more details.", "Feedback mismatch."

        mock_completion.assert_called_once_with(
            messages=[{"content": expected_message, "role": "user"}], **request_settings
        )


def test_generate_improved_version():
    """
    Test generate_improved_version with mocked structured response.
    """
    prompt = "What is the capital of France?"
    answer = "Paris"
    feedback = "Add more details about France."
    request_settings = {"model_name": "gpt-3.5-turbo"}

    mock_improved_response = MockImprovedResponse(
        ImprovedText="The capital of France is Paris, located in Europe."
    )

    with patch(
        "llm_mcts_inference.inference.get_structured_model_response",
        return_value=mock_improved_response,
    ):
        response = generate_improved_version(
            prompt, answer, feedback, request_settings, MockImprovedResponse
        )

        assert (
            response == "The capital of France is Paris, located in Europe."
        ), "Improved version mismatch."


def test_get_model_response():
    """
    Test get_model_response with mocked response.
    """
    prompt = "What is the capital of France?"
    request_settings = {"model_name": "gpt-3.5-turbo"}

    mock_response = {"choices": [{"message": {"content": "The capital of France is Paris."}}]}

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        response = get_model_response(prompt, request_settings)

        assert response == "The capital of France is Paris.", "Response content mismatch."
        mock_completion.assert_called_once_with(
            messages=[{"content": prompt, "role": "user"}], **request_settings
        )


def test_get_structured_model_response():
    """
    Test get_structured_model_response with mocked structured response.
    """
    prompt = "What is the capital of France?"
    request_settings = {"model_name": "gpt-3.5-turbo"}
    mock_schema = MockImprovedResponse

    mock_structured_response = MockImprovedResponse(
        ImprovedText="The capital of France is Paris, located in Europe."
    )

    with patch("instructor.from_litellm") as mock_instructor:
        mock_client = mock_instructor.return_value
        mock_client.create.return_value = mock_structured_response

        response = get_structured_model_response(prompt, request_settings, mock_schema)

        assert response == mock_structured_response, "Structured response mismatch."

        mock_client.create.assert_called_once_with(
            response_model=mock_schema,
            messages=[{"content": prompt, "role": "user"}],
            **request_settings,
        )
