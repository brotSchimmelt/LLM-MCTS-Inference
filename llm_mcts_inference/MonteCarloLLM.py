import os
from dataclasses import dataclass
from typing import Any, List

from dotenv import load_dotenv


@dataclass
class MCTSResult:
    answer: str
    tree: Any  # TODO: Define tree type
    valid_path: List[Any]  # TODO: Define tree type


class MonteCarloLLM:
    def __init__(self, model_name: str, endpoint: str = "", api_key: str = "") -> None:
        load_dotenv()

        self.model_name = model_name
        self.endpoint = endpoint if endpoint else "https://api.openai.com"
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "EMPTY"

        if not model_name:
            raise ValueError("Please provide a model name.")

    def generate(self, prompt: str) -> MCTSResult:
        """Generates a answer using Monte Carlo Tree Search with the given prompt.

        Args:
            prompt (str): The input prompt or query.

        Returns:
            MCTSResult: An object containing the generated answer, the complete MCTS tree,
            and the valid path representing the sequence of decisions leading to the answer.
        """
        raise NotImplementedError("Implement me!")
        return None

    def __str__(self) -> str:
        return f"MonteCarloLLM(model_name={self.model_name}, endpoint={self.endpoint})"

    def __repr__(self) -> str:
        return str(self)
