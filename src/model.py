from config import BASE_MODEL, MAX_SEQ_LEN, LORA_R

import torch
import gc
from unsloth import FastLanguageModel

def ClearGPU():
    gc.collect()
    torch.cuda.empty_cache()
    print(f'    ✅ Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

def ModelFactory():
    ClearGPU()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True, # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.05,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=42,
    )

    model.print_trainable_parameters()

    print(f'    ✅ PEFT and QLoRa ready')
    print(f'    ✅ Model ready')

    return model, tokenizer
