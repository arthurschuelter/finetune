from dataclasses import dataclass, field

@dataclass
class ModelfileConfig:
    system_prompt: str
    temperature: float = 0.7
    num_ctx: int = 4096
    num_predict: int = 512
    repeat_penalty: float = 1.3
    model_filename: str = "apereal-model-v1",
    model_dir: str = "./model",

    stop_tokens: list[str] = field(default_factory=lambda: [
        "<|im_end|>", "<|endoftext|>", "Usuário:", "User:", "\n\n\n"
    ])

    def render(self) -> str:
        stops = "\n".join(f'PARAMETER stop "{t}"' for t in self.stop_tokens)
        return f"""\
            FROM ./{self.model_filename}.gguf
            SYSTEM \"\"\"{self.system_prompt}\"\"\"
            PARAMETER temperature {self.temperature}
            PARAMETER num_ctx {self.num_ctx}
            PARAMETER num_predict {self.num_predict}
            PARAMETER repeat_penalty {self.repeat_penalty}
            {stops}
        """

    def export(self) -> None:
        content = self.render()
        path = f"{self.model_dir}/Modelfile"
        with open(path, "w") as f:
            f.write(content)
        print("✅ Modelfile created successfully!")
