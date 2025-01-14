import numpy as np

DEFAULT_SETTINGS = {
    "model": "gpt-3.5-turbo",
    "base_url": "https://api.openai.com/v1",
    "api_key": "EMPTY",
    "max_children": 3,
    "exploration_weight": np.sqrt(2),
    "iterations": 10,
    "verbose": True,
}

MODEL_SETTINGS = {
    "max_tokens": 8_192,
    "temperature": 1.0,
    "top_p": 0.9,
    "seed": 1337,
}
