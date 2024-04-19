# load jamba from HF
# inference

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

prompt = "def min(arr):"

tokenizer = AutoTokenizer.from_pretrained(
    "TechxGenus/Mini-Jamba",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

print("tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    "TechxGenus/Mini-Jamba",
    torch_dtype=torch.float16,
    device_map="auto", # cuda
    trust_remote_code=True,
    use_mamba_kernels=False
)

print("model loaded")

inputs = tokenizer.encode(prompt, return_tensors="pt")

st = time.time()
outputs = model.generate(
    input_ids=inputs.to(model.device),
    max_new_tokens=512,
    do_sample=False,
)

et = time.time()

print(et-st)
print(len(outputs[0]))
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)
