def LoadAperealDataset():
    import json
    from datasets import Dataset

    path = "/mnt/c/Users/Arthu/Documents/Study/finetune/src/data/dataset.json"
    dataset = []

    with open(path) as f:
        dataset = json.load(f)

    # dataset = dataset['train']
    dataset = Dataset.from_list(dataset["train"])

    print("Dataset succesfully loaded:")
    print("size: ", len(dataset))

    return dataset

def LoadTest():
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

