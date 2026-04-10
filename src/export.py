def ExportModefile(model_filename, file_path):

    # TODO: This should be more flexible, with parameters and system instructions being passed in as arguments, but for now it's hardcoded for the specific use case of the ApeReal chatbot.
    # TODO: Create a prompt module
    text_string = f'''
    
    FROM ./{model_filename}.gguf
    SYSTEM """você é um atendente da ApeReal, e irá tirar duvidas sobre o programa Minha casa Minha vida, um programa social do goveno que auxilia na obtenção de uma casa prória"""
    SYSTEM """...Responda apenas uma vez. Não continue o texto após dar a resposta."""
    PARAMETER temperature 0.7
    PARAMETER num_ctx 4096
    PARAMETER num_predict 512
    PARAMETER repeat_penalty 1.3
    PARAMETER stop "<|im_end|>"
    PARAMETER stop "<|endoftext|>"
    PARAMETER stop "Usuário:"
    PARAMETER stop "User:"
    PARAMETER stop "\n\n\n"
    '''

    with open(file_path + "/Modelfile", "w") as f:
        f.write(text_string)
    print("✅ Modelfile created successfully!")
