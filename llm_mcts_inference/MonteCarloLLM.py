import os
from dataclasses import dataclass
from typing import Any, Dict, List

from config import DEFAULT_SETTINGS, MODEL_SETTINGS
from dotenv import load_dotenv


@dataclass
class MCTSResult:
    answer: str
    tree: Any  # TODO: Define tree type
    valid_path: List[Any]  # TODO: Define tree type


class MonteCarloLLM:
    def __init__(
        self,
        model_name: str,
        endpoint: str = "",
        api_key: str = "",
        model_settings: Dict[str, Any] = {},
    ) -> None:
        load_dotenv()

        self.model_name = model_name if model_name else DEFAULT_SETTINGS["model"]
        self.endpoint = endpoint if endpoint else DEFAULT_SETTINGS["base_url"]
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_SETTINGS["api_key"]

        self.model_settings = self._validate_model_settings(model_settings)

        self.model_settings["model_name"] = self.model_name
        self.model_settings["base_url"] = self.endpoint
        self.model_settings["api_key"] = self.api_key

    def _validate_model_settings(self, model_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and updates the provided model settings against the default configuration.

        Args:
            model_settings (Dict[str, Any]): A dictionary of user-provided model settings
                to validate and update. The keys in this dictionary should match the
                expected keys defined in `MODEL_SETTINGS`.

        Returns:
            Dict[str, Any]: A validated dictionary of model settings. This includes all
                the default values from `MODEL_SETTINGS`, updated with any valid keys
                and values provided in the `model_settings` argument.
        """
        valid_settings: Dict[str, Any] = MODEL_SETTINGS.copy()

        for key, value in model_settings.items():
            if key in valid_settings:
                valid_settings[key] = value

        return valid_settings

    def generate(self, prompt: str) -> MCTSResult:
        raise NotImplementedError("Implement me!")
        return None

    def __str__(self) -> str:
        return f"MonteCarloLLM(model_settings={self.model_settings}"

    def __repr__(self) -> str:
        return str(self)
