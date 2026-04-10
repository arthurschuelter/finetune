MODEL_REGISTRY: dict[str, dict] = {
    "Qwen3-4B": {
        "path": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    },
    "Gemma3-4B": {
        "path": "unsloth/gemma-3-4b-it-bnb-4bit",
    },
    "Gemma3-4B-Dynamic": {
        "path": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    },
    "Llama3.2-3B": {
        "path": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    },
    "Llama3.2-3B-Dynamic": {
        "path": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    },
}


def resolve(model_name: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Model "{model_name}" not found in registry')
    return MODEL_REGISTRY[model_name]
