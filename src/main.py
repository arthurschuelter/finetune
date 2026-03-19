## ✅ Step 0 — Verify GPU
import torch
print(f'\nPyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
print(f'GPU             : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

## ⚙️ Step 2 — Configuration
print('⚙️ Step 2 — Configuration')

# ── Tweak these to your needs ──────────────────────────────────────────────
BASE_MODEL = 'unsloth/tinyllama-bnb-4bit'  # correct name
  # fits T4 free tier
# Want a 7B? Use 'unsloth/mistral-7b-v0.3-bnb-4bit' but needs Colab Pro (A100)
OUTPUT_DIR  = './qlora-output'
MAX_SEQ_LEN = 512
EPOCHS      = 3
BATCH_SIZE  = 2
GRAD_ACCUM  = 8    # effective batch = 2 * 8 = 16
LORA_R      = 16
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
from datasets import load_dataset, concatenate_datasets

def format_alpaca(row):
    instruction = (row.get('instruction') or '').strip()
    ctx         = (row.get('input') or '').strip()
    output      = (row.get('output') or '').strip()
    prompt = f'{instruction}\n{ctx}' if ctx else instruction
    return {'text': f'<s>[INST] {prompt} [/INST] {output}</s>'}

def format_generic(row):
    prompt = (row.get('instruction') or row.get('prompt') or row.get('question') or '').strip()
    answer = (row.get('output') or row.get('response') or row.get('completion') or row.get('answer') or '').strip()
    ctx    = (row.get('input') or row.get('context') or '').strip()
    if ctx:
        prompt = f'{prompt}\n{ctx}'
    return {'text': f'<s>[INST] {prompt} [/INST] {answer}</s>'}

parts = []

# ── CodeAlpaca-20k ─────────────────────────────────────────────────────────
print(f'    ✅ Loading CodeAlpaca...')
coding = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
coding_fmt = coding.map(format_alpaca, remove_columns=coding.column_names)
parts.append(coding_fmt)
print(f'  CodeAlpaca: {len(coding_fmt):,} examples')

# ── CyberNative Cybersecurity ──────────────────────────────────────────────
try:
    print(f'    ✅ Loading CyberNative...')
    vulns = load_dataset('CyberNative-AI/Cybersecurity_Specialized_Dataset', split='train')
    print(f'    ✅ Columns: {vulns.column_names}')
    vulns_fmt = vulns.map(format_generic, remove_columns=vulns.column_names)
    parts.append(vulns_fmt)
    print(f'    ✅ CyberNative: {len(vulns_fmt):,} examples')
except Exception as e:
    print(f'    ❌ Skipped CyberNative: {e}')

# ── CVE explanations ───────────────────────────────────────────────────────
try:
    print(f'    ✅ Loading CVE dataset...')
    cve = load_dataset('detomo/cve-explain-openai', split='train')
    print(f'    ✅ Columns: {cve.column_names}')
    cve_fmt = cve.map(format_generic, remove_columns=cve.column_names)
    parts.append(cve_fmt)
    print(f'    ✅ CVE: {len(cve_fmt):,} examples')
except Exception as e:
    print(f'    ❌ Skipped CVE: {e}')

# ── Combine & clean ────────────────────────────────────────────────────────
dataset = concatenate_datasets(parts).shuffle(seed=42)
dataset = dataset.filter(lambda x: len(x['text'].strip()) > 30)

print()
print(f'    ✅ Total training examples: {len(dataset):,}')
print(f'    ✅ Sample:\n{dataset[0]["text"][:400]}')


## 🚀 Step 5 — Train
print('🚀 Step 5 — Train')
from trl import SFTTrainer, SFTConfig

dataset = dataset.select(range(500))
print(f'Using {len(dataset)} examples')

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
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
print()
print(f'✅ Training complete! Adapter saved to {OUTPUT_DIR}')


# ## 🔗 Step 6 — Merge & Save (16-bit)
# MERGED_DIR = './merged-model'

# # Unsloth handles the merge natively — no manual PeftModel needed
# model.save_pretrained_merged(
#     MERGED_DIR,
#     tokenizer,
#     save_method='merged_16bit',
# )
# print(f'✅ Merged model saved to {MERGED_DIR}')

# ## 🔄 Step 8 — Export to GGUF
# model.save_pretrained_gguf(
#     'code-sec-model',
#     tokenizer,
#     quantization_method='q4_k_m',
# )

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