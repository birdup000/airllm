from airllm import AutoModel
import time
from airllm.utils import clean_memory

MAX_LENGTH = 128

model = AutoModel.from_pretrained('nvidia/Llama-3.1-Nemotron-70B-Instruct-HF')

input_text = ['What is the capital of United States?']

input_tokens = model.tokenizer(input_text, return_tensors='pt', return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=False)

# Start profiling
model.profiler.add_profiling_time('start_inference', time.time())

generation_output = model.generate(input_tokens['input_ids'].cuda(), max_new_tokens=20, use_cache=True, return_dict_in_generate=True)

# End profiling
model.profiler.add_profiling_time('end_inference', time.time())
model.profiler.print_profiling_time()
clean_memory()

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
