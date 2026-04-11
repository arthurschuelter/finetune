def main():
    # from utils.environment import CheckGPU
    # CheckGPU()

    print("Step 1 — Configuration")
    from .config import (
        BASE_MODEL,
        BATCH_SIZE,
        EPOCHS,
        GRAD_ACCUM,
        LORA_R,
        MAX_SEQ_LEN,
        MERGED_DIR,
        MODEL_FILENAME,
        MODEL_PROMPT,
        OUTPUT_DIR,
    )

    print(f"    ✅ Base model: {BASE_MODEL}")
    print(f"    ✅ Output directory: {OUTPUT_DIR}")
    print(f"    ✅ Maximum sequence length: {MAX_SEQ_LEN}")
    print(f"    ✅ Epochs: {EPOCHS}")
    print(f"    ✅ Batch size: {BATCH_SIZE}")
    print(f"    ✅ Gradient accumulation: {GRAD_ACCUM}")
    print(f"    ✅ LoRA rank: {LORA_R}")
    print("    ✅ Config ready")

    print("Step 2 — Load Model")
    from .models.model import Model
    from .models.model_factory import ModelFactory

    _model: Model = ModelFactory(BASE_MODEL)
    model, tokenizer = _model.model, _model.tokenizer

    print("Step 3 — Load & Format Datasets")
    from .load_dataset import LoadAperealDataset, formatting_func

    dataset = LoadAperealDataset()

    print("Step 4 — Train")
    from .train import trainModel

    model, tokenizer = trainModel(model, dataset, tokenizer, formatting_func)

    print("Step 5 — Merge & Save")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✅ Merged model saved to {MERGED_DIR}")

    print("Step 6 — Export to GGUF")
    model.save_pretrained_gguf(
        MODEL_FILENAME,
        tokenizer,
        quantization_method="q4_k_m",
    )

    import glob

    gguf_files = glob.glob(MODEL_FILENAME + "*.gguf")
    print(f"✅ GGUF file(s): {gguf_files}")

    from .prompt.modelfile import ModelfileConfig
    _modelfile_config = ModelfileConfig(
        system_prompt=MODEL_PROMPT,
        model_filename=MODEL_FILENAME,
        model_dir=MERGED_DIR
    )

    _modelfile_config.export()

    # from .export import ExportModefile
    # ExportModefile(MODEL_FILENAME, MERGED_DIR)


if __name__ == "__main__":
    main()
