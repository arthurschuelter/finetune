## ✅ Step 0 — Verify GPU
import torch
print(f'\nPyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
print(f'GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

## ⚙️ Step 2 — Configuration
print('⚙️ Step 2 — Configuration')

# ── Models ────────────────────────────────────────────────────────────────────
# BASE_MODEL = 'unsloth/tinyllama-bnb-4bit'

# ✅ Qwen3-4B — best overall, native PT-BR, 128K context, has "thinking mode"
# BASE_MODEL = 'unsloth/Qwen3-4B-bnb-4bit'
# # Dynamic 2.0 variant (recommended):
BASE_MODEL = 'unsloth/Qwen3-4B-unsloth-bnb-4bit'

# # ✅ Gemma 3 4B — multimodal, 140+ languages, 128K context
# BASE_MODEL = 'unsloth/gemma-3-4b-it-bnb-4bit'
# # Dynamic 2.0 variant:
# BASE_MODEL = 'unsloth/gemma-3-4b-it-unsloth-bnb-4bit'

# # ✅ Llama 3.2 3B — PT officially supported, very tuneable
# BASE_MODEL = 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'
# # Dynamic 2.0 variant:
# BASE_MODEL = 'unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit'

# # ⚠️  Amadeus-Verbo — NOT on Unsloth hub, load directly from HF
# BASE_MODEL = 'nicholasKluge/Aira-2-portuguese-7B'   # older, well-known PT-BR option
# # or the Amadeus family:
# BASE_MODEL = 'dominguesm/Amadeus-Verbo-7B'

MODEL_FILENAME = 'apereal-' + BASE_MODEL + '-v1'
MODEL_FILENAME = MODEL_FILENAME.replace('/', '-')
print('Model -> ', MODEL_FILENAME)
OUTPUT_DIR  = './qlora-output'
MAX_SEQ_LEN = 512
EPOCHS      = 3     # Depends on eval_loss, increasing eval_loss leads to over-
                    # fitting
BATCH_SIZE  = 1     # Higher => Faster
GRAD_ACCUM  = 8     # effective batch = 2 * 8 = 16
LORA_R      = 16    # Maybe 64?
# ───────────────────────────────────────────────────────────────────────────
print(f'    ✅ Config ready')

## 🤖 Step 3 — Load Model
print('🤖 Step 3 — Load Model')

import gc
from unsloth import FastLanguageModel

gc.collect()
torch.cuda.empty_cache()
print(f'    ✅ Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True, # QLoRA
)
print(f'    ✅ Pretrained model ready')


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
print(f'    ✅ PEFT and QLoRa ready')


model.print_trainable_parameters()
print(f'    ✅ Model ready')

## 🗄️ Step 4 — Load & Format Datasets
print('🗄️ Step 4 — Load & Format Datasets')
from load_dataset import LoadAperealDataset, LoadTest

dataset = LoadAperealDataset()
# dataset = LoadTest()

def formatting_func(example):
    outputs = []

    for messages in example["messages"]:
        # Case: string
        if isinstance(messages, str):
            text = messages.strip()
            if text:
                outputs.append(text)
            continue

        # Case: list of dicts
        text = ""
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "").strip()

                if not content:
                    continue

                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"

        if text.strip():
            outputs.append(text.strip())

    if not outputs:
        return [""]  # or [" "] if needed

    return outputs


## 🚀 Step 5 — Train
print('🚀 Step 5 — Train')
from trl import SFTTrainer, SFTConfig

dataset = dataset.select(range(50))
print(f'Using {len(dataset)} examples')

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=50,
        save_strategy='epoch',
        save_total_limit=2,
        optim='adamw_8bit',
        lr_scheduler_type='cosine',
        warmup_steps=100,
        report_to='none',
        dataset_text_field='text',
        max_seq_length=MAX_SEQ_LEN,
    ),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'\n✅ Training complete! Adapter saved to {OUTPUT_DIR}')


## 🔗 Step 6 — Merge & Save (16-bit)
MERGED_DIR = './merged-model'

# Unsloth handles the merge natively — no manual PeftModel needed
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method='merged_16bit',
)
print(f'✅ Merged model saved to {MERGED_DIR}')

## 🔄 Step 8 — Export to GGUF
model.save_pretrained_gguf(
    MODEL_FILENAME,
    tokenizer,
    quantization_method='q4_k_m',
)

import glob
gguf_files = glob.glob(MODEL_FILENAME + '*.gguf')
print(f'✅ GGUF file(s): {gguf_files}')

# import glob
# gguf_files = glob.glob('code-sec-model*.gguf')
# print(f'✅ GGUF file(s): {gguf_files}')

# ## 🖥️ Step 9 — Import into Ollama (run on your local machine)
# # cat > Modelfile <<'EOF'
# # FROM ./code-sec-model-unsloth.Q4_K_M.gguf
# # SYSTEM """You are an expert software engineer and security researcher.
# # You write clean, correct code and proactively identify vulnerabilities,
# # explain their root cause, and suggest secure alternatives.
# # When reviewing code, always check for: injection flaws, buffer overflows,
# # insecure deserialization, broken authentication, hardcoded secrets,
# # and OWASP Top 10 issues."""
# # PARAMETER temperature 0.2
# # PARAMETER num_ctx 4096
# # EOF

# # ollama create code-sec-model -f Modelfile
# # ollama run code-sec-model