import os

# ── Models ────────────────────────────────────────────────────────────────────
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

DATASET_SIZE = 1000
MERGED_DIR = './merged-model'
# ───────────────────────────────────────────────────────────────────────────
