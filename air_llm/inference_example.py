from airllm import AutoModel

MAX_LENGTH = 128


model = AutoModel.from_pretrained("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )

input_text = [
        'What is the capital of United States?',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=10,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
