from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from torch.cuda.amp import autocast

# Load the model and tokenizer
model_name = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use mixed precision
MAX_LENGTH = 128

input_text = ['What is the capital of United States?']

# Efficient tokenization
input_tokens = tokenizer(input_text, return_tensors='pt', return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=False)

# Batch processing
batch_size = 8
input_ids = input_tokens['input_ids'].repeat(batch_size, 1).cuda()

# Start profiling
start_time = time.time()

with autocast():
    generation_output = model.generate(input_ids, max_new_tokens=20, use_cache=True, return_dict_in_generate=True)

# End profiling
end_time = time.time()
print(f"Inference time: {end_time - start_time} seconds")

output = tokenizer.decode(generation_output.sequences[0])

print(output)


