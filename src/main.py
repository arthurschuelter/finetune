## ✅ Step 0 — Verify GPU
# from utils.environment import CheckGPU
# CheckGPU()

print('Step 1 — Configuration')
from config import BASE_MODEL, OUTPUT_DIR, MAX_SEQ_LEN, EPOCHS, BATCH_SIZE, GRAD_ACCUM, LORA_R, MODEL_FILENAME, MERGED_DIR

print(f'    ✅ Base model: {BASE_MODEL}')
print(f'    ✅ Output directory: {OUTPUT_DIR}')
print(f'    ✅ Maximum sequence length: {MAX_SEQ_LEN}')
print(f'    ✅ Epochs: {EPOCHS}')
print(f'    ✅ Batch size: {BATCH_SIZE}')
print(f'    ✅ Gradient accumulation: {GRAD_ACCUM}')
print(f'    ✅ LoRA rank: {LORA_R}')
print(f'    ✅ Config ready')

print('Step 2 — Load Model')
from model import ModelFactory
model, tokenizer = ModelFactory()

print('Step 3 — Load & Format Datasets')
from load_dataset import LoadAperealDataset, LoadTest, formatting_func

dataset = LoadAperealDataset()

print('Step 4 — Train')
from train import trainModel
model, tokenizer = trainModel(model, dataset, tokenizer, formatting_func)


print('Step 5 — Merge & Save')
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method='merged_16bit',
)
print(f'✅ Merged model saved to {MERGED_DIR}')

print('Step 6 — Export to GGUF')
model.save_pretrained_gguf(
    MODEL_FILENAME,
    tokenizer,
    quantization_method='q4_k_m',
)

import glob
gguf_files = glob.glob(MODEL_FILENAME + '*.gguf')
print(f'✅ GGUF file(s): {gguf_files}')

from export import ExportModefile
ExportModefile(MODEL_FILENAME, MERGED_DIR)

# ## 🖥️ Step 7 — Import into Ollama (run on your local machine)
# # cat > Modelfile <<'EOF'
# # FROM ./code-sec-model-unsloth.Q4_K_M.gguf
# # SYSTEM """You are an expert software engineer and security researcher.
# # You write clean, correct code and proactively identify vulnerabilities,
# # explain their root cause, and suggest secure alternatives.
# # PARAMETER temperature 0.2
# # PARAMETER num_ctx 4096
# # EOF

# # ollama create code-sec-model -f Modelfile
# # ollama run code-sec-model
