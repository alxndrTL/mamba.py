# load jamba from HF
# inference w/ 50 tokens

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

prompt = "def min(arr):"

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

print("tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1", device_map="cuda", trust_remote_code=True, use_mamba_kernels=False)

print("model loaded")

input_ids = tokenizer("In the recent Super Bowl LVIII,", return_tensors='pt').to(model.device)["input_ids"]

st = time.time()
outputs = model.generate(input_ids, max_new_tokens=216)
et = time.time()

print(et-st)
print(len(outputs[0]))
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)
