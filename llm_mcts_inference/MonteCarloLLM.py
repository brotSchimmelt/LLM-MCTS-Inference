from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

from .config import DEFAULT_REQUEST_SETTINGS, DEFAULT_SETTINGS
from .mcts import MCTS, Node


@dataclass
class MCTSResult:
    """
    Represents the result of a Monte Carlo Tree Search (MCTS) operation.

    Attributes:
        answer (str): The answer derived from the MCTS process.
        tree (TreeNode): The root node of the MCTS tree.
        valid_path (List[Node]): The sequence of valid nodes leading to the result.
    """

    answer: str
    tree: Node
    valid_path: List[Node]


class MonteCarloLLM:
    """
    A class to interface with Monte Carlo Tree Search (MCTS) for a language model (LLM).

    Attributes:
        model_name (str): The name of the model being used.
        api_base (str): The base URL for the API endpoint.
    """

    def __init__(self, model_name: str = DEFAULT_SETTINGS["model"], api_base: str = "") -> None:
        load_dotenv()

        self.model_name: str = model_name if model_name else DEFAULT_SETTINGS["model"]
        self.api_base = api_base

        if not self.api_base and "ollama" in model_name:
            self.api_base = DEFAULT_SETTINGS["ollama_api_base"]

    def generate(
        self,
        prompt: str,
        request_settings: Dict[str, Any] = DEFAULT_REQUEST_SETTINGS,
        iterations: int = DEFAULT_SETTINGS["iterations"],
        max_children: int = DEFAULT_SETTINGS["max_children"],
        verbose: bool = DEFAULT_SETTINGS["verbose"],
        exploration_weight: float = DEFAULT_SETTINGS["exploration_weight"],
    ) -> MCTSResult:
        """
        Generates a response using Monte Carlo Tree Search (MCTS).

        Args:
            prompt (str): The input prompt for the language model.
            request_settings (Dict[str, Any], optional): Settings for the request to the language
                model API. Defaults to DEFAULT_REQUEST_SETTINGS.
            iterations (int, optional): The number of iterations for the MCTS process. Defaults to
                DEFAULT_SETTINGS["iterations"].
            max_children (int, optional): The maximum number of child nodes to expand per node.
                Defaults to DEFAULT_SETTINGS["max_children"].
            verbose (bool, optional): Flag for verbose logging. Defaults to
                DEFAULT_SETTINGS["verbose"].
            exploration_weight (float, optional): The exploration weight used in the MCTS algorithm.
                Defaults to DEFAULT_SETTINGS["exploration_weight"].

        Returns:
            MCTSResult: The result of the MCTS process, including the answer, the valid path, and
                the MCTS tree.
        """
        request_settings["api_base"] = self.api_base
        request_settings["model"] = self.model_name

        # create MCTS tree
        mcts_tree: MCTS = MCTS(
            original_prompt=prompt.strip(),
            request_settings=request_settings,
            iterations=iterations,
            max_children=max_children,
            verbose=verbose,
            exploration_weight=exploration_weight,
        )

        return MCTSResult(
            answer=mcts_tree.search(),
            valid_path=mcts_tree.get_best_path(),
            tree=mcts_tree.get_tree(),
        )

    def __str__(self) -> str:
        if self.api_base:
            return f"MonteCarloLLM(model={self.model_name}, api_base={self.api_base})"
        return f"MonteCarloLLM(model={self.model_name})"

    def __repr__(self) -> str:
        return str(self)
