from config import OUTPUT_DIR, MAX_SEQ_LEN, EPOCHS, BATCH_SIZE, GRAD_ACCUM, DATASET_SIZE
from trl import SFTTrainer, SFTConfig


def trainModel(model, dataset, tokenizer, formatting_func):
    dataset = dataset.select(range(DATASET_SIZE))

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

    return model, tokenizer