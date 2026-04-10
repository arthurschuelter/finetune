import gc

import torch
from unsloth import FastLanguageModel

from ..config import LORA_R, MAX_SEQ_LEN
from .model import Model
from .model_registry import resolve


def _clear_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"    ✅ Free VRAM: {free_gb:.1f} GB")


def _load_base_model(model_name: str) -> tuple[any, any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    return model, tokenizer


def _apply_peft(model: any) -> any:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model


# TODO: Modularizar o modelo + prompt
#       - criar um módulo de prompt engineering, onde cada modelo tem um prompt específico, e a factory retorna o prompt correto junto com o modelo e tokenizer
def ModelFactory(model_name: str) -> Model:
    _clear_gpu()
    base_model = resolve(model_name)

    model, tokenizer = _load_base_model(base_model["path"])
    model = _apply_peft(model)
    print("    ✅ PEFT and QLoRa ready")

    model.print_trainable_parameters()

    return Model(model, tokenizer)
