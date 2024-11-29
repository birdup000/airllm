from airllm import AutoModel
import torch

MAX_LENGTH = 128

# Initialize model
model = AutoModel.from_pretrained(
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    compression='4bit',
    device="cuda:0"
)

# Set pad token to eos token if not set
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token

input_text = [
    'Tell me about creatine',
]

# Tokenize with padding
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH
)

# Generate with attention mask
with torch.cuda.stream(model.stream):
    generation_output = model.generate(
        input_ids=input_tokens['input_ids'].cuda(),
        attention_mask=input_tokens['attention_mask'].cuda(),
        max_new_tokens=64,
        use_cache=True,
        return_dict=True,
        decoding_strategy="jacobi",
        decoding_kwargs={
            "n_gram_size": 4,
            "max_iterations": 10
        }
    )

torch.cuda.current_stream().synchronize()
output = model.tokenizer.decode(generation_output.sequences[0])
print(output)