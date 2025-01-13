from typing import Any, Dict, cast

import openai
import outlines
from pydantic import BaseModel

from .prompts import ImprovedResponse, RatingResponse, critique_prompt, rating_prompt, refine_prompt
from .utils import normalize_rating_score


def generate_initial_answer(prompt: str, model_settings: Dict[str, Any]) -> str:
    """
    Generates the initial answer using greedy decoding.

    Args:
        prompt (str): The input prompt to guide the model's response.
        model_settings (Dict[str, Any]): A dictionary containing configuration settings
            for the model, including temperature, top_p, and other parameters.

    Returns:
        str: The initial response from the model as a string.
    """
    # greedy decoding parameters
    model_settings["temperature"] = 0.0
    model_settings["top_p"] = 1.0

    return get_model_response(prompt, model_settings)


def generate_rating(
    prompt: str, answer: str, model_settings: Dict[str, Any], rating_schema: BaseModel
) -> float:
    """
    Generates a normalized rating score for a given answer based on a prompt and schema.

    Args:
        prompt (str): The original input prompt to guide the model's response.
        answer (str): The generated or improved answer to be rated.
        model_settings (Dict[str, Any]): A dictionary containing configuration settings
            for the model, such as API key, model name, and decoding parameters.
        rating_schema (BaseModel): A Pydantic schema that defines the structure of the
            expected rating response.

    Returns:
        float: A normalized rating score within the range [0, 0.95].
    """
    rating_response = get_structured_model_response(
        rating_prompt.format(original_prompt=prompt, improved_answer=answer),
        model_settings=model_settings,
        json_schema=RatingResponse,
    )
    rating = rating_response.rating

    # check type of the rating
    if not isinstance(rating, (int, str)):
        rating = str(rating)

    # normalize the rating to be within the range [0, 0.95]
    return normalize_rating_score(rating)


def generate_feedback(prompt: str, answer: str, model_settings: Dict[str, Any]) -> str:
    raise NotImplementedError("Implement me!")
    return ""


def generate_improved_version(
    prompt: str, answer: str, feedback: str, model_settings: Dict[str, Any]
) -> str:
    raise NotImplementedError("Implement me!")
    return ""


def get_model_response(prompt: str, model_settings: Dict[str, Any]) -> str:
    """
    Generates a response from an LLM based on the provided prompt and model settings.

    Args:
        prompt (str): The input prompt to guide the model's response.
        model_settings (Dict[str, Any]): A dict containing configuration settings for the model.
            Expected keys include:
            - "base_url" (str): The API endpoint for the LLM.
            - "api_key" (str): The API key for authenticating requests.
            - "model_name" (str): The name of the LLM to use.
            - "max_tokens" (int): The maximum number of tokens to generate in the response.
            - "temperature" (float): The temperature for response sampling (controls randomness).
            - "top_p" (float): The nucleus sampling parameter for response diversity.
            - "seed" (int): A seed value for deterministic responses.

    Returns:
        str: The generated text response from the LLM.
    """
    client = openai.OpenAI(
        base_url=str(model_settings["base_url"]),
        api_key=str(model_settings["api_key"]),
    )

    response = client.completions.create(
        model=str(model_settings["model_name"]),
        prompt=prompt,
        max_tokens=int(model_settings["max_tokens"]),
        temperature=float(model_settings["temperature"]),
        top_p=float(model_settings["top_p"]),
        seed=int(model_settings["seed"]),
    )

    return str(response.choices[0].text)


def get_structured_model_response(
    prompt: str, model_settings: Dict[str, Any], json_schema: Any
) -> Any:
    """
    Generates a structured response using an LLM based on the provided prompt and schema.

    Args:
        prompt (str): The input prompt to guide the model's response.
        model_settings (Dict[str, Any]): A dict containing configuration settings for the model.
            Expected keys include:
            - "model_name" (str): The name of the LLM to use.
            - "base_url" (str): The API endpoint for the LLM.
            - "api_key" (str): The API key for authenticating requests.
            - "max_tokens" (int): The maximum number of tokens to generate in the response.
            - "temperature" (float): The temperature for response sampling (controls randomness).
            - "top_p" (float): The nucleus sampling parameter for response diversity.
        json_schema (Any): A Pydantic model that defines the expected structure of the
            response.

    Returns:
        Any: The structured response generated by the LLM, conforming to the given schema.
    """
    model = model_settings["model_name"]
    base_url = model_settings["base_url"]
    api_key = model_settings["api_key"]

    # define the llm to use
    llm = outlines.models.openai(model, base_url=base_url, api_key=api_key)  # type: ignore[attr-defined]

    # define the sampler we want to use
    sampler = cast(
        outlines.samplers.Sampler,
        outlines.samplers.MultinomialSampler(
            temperature=float(model_settings["temperature"]), top_p=float(model_settings["top_p"])
        ),
    )

    generator = outlines.generate.json(llm, json_schema, sampler)  # type: ignore[attr-defined]
    return generator(prompt, max_tokens=int(model_settings["max_tokens"]))
