import os

from dotenv import load_dotenv


class MonteCarloLLM:
    def __init__(
        self, model_name: str, endpoint: str = None, api_key: str = None
    ) -> None:
        load_dotenv()

        self.model_name = model_name
        self.endpoint = endpoint if endpoint else "https://api.openai.com"
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "EMPTY"

        if not model_name:
            raise ValueError("Please provide a model name.")

    def __str__(self) -> str:
        return f"MonteCarloLLM(model_name={self.model_name}, endpoint={self.endpoint})"

    def __repr__(self) -> str:
        return str(self)
