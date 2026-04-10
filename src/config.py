import os

# ── Models ────────────────────────────────────────────────────────────────────
BASE_MODEL = 'Qwen3-4B'
# BASE_MODEL = 'Gemma3-4B'
# BASE_MODEL = 'Llama3.2-3B'
# BASE_MODEL = 'Gemma3-4B-Dynamic'
# BASE_MODEL = 'Llama3.2-3B-Dynamic'
# ───────────────────────────────────────────────────────────────────────────

MODEL_FILENAME = 'apereal-' + BASE_MODEL + '-v1'
MODEL_FILENAME = MODEL_FILENAME.replace('/', '-')
print('Model -> ', MODEL_FILENAME)
OUTPUT_DIR  = './qlora-output'
MAX_SEQ_LEN = 512
EPOCHS      = 3     # Depends on eval_loss, increasing eval_loss leads to over-
                    # fitting
BATCH_SIZE  = 1     # Higher => Faster
GRAD_ACCUM  = 8      
LORA_R      = 16    # Maybe 64?

DATASET_SIZE = 1000
MERGED_DIR = './merged-model'
# ───────────────────────────────────────────────────────────────────────────
