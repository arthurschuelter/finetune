import os

# ── Models ────────────────────────────────────────────────────────────────────
# Qwen3-4B 
BASE_MODEL = 'unsloth/Qwen3-4B-unsloth-bnb-4bit'

# # Gemma 3 4B
# BASE_MODEL = 'unsloth/gemma-3-4b-it-bnb-4bit'
# # Dynamic 2.0 variant:
# BASE_MODEL = 'unsloth/gemma-3-4b-it-unsloth-bnb-4bit'

# #  Llama 3.2 3B
# BASE_MODEL = 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit'
# # Dynamic 2.0 variant:
# BASE_MODEL = 'unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit'
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
