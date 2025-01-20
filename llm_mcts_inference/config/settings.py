import numpy as np

DEFAULT_SETTINGS = {
    "model": "openai/gpt-4o-mini",
    "ollama_api_base": "http://localhost:11434",
    "api_key": "EMPTY",
    "max_children": 3,
    "exploration_weight": np.sqrt(2),
    "iterations": 10,
    "verbose": True,
}

DEFAULT_REQUEST_SETTINGS = {
    "max_tokens": 8_192,
    "temperature": 1.0,
    "top_p": 0.9,
    "seed": 1337,
}
