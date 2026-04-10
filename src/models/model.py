class Model:
    def __init__(self, model, tokenizer, prompt=''):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        
        print(f'    ✅ Model ready')
