# load jamba from HF
# inference w/ 50 tokens

import torch
from jamba import from_pretrained
from transformers import AutoTokenizer

import time

prompt = "def min(arr):"

tokenizer = AutoTokenizer.from_pretrained(
    "TechxGenus/Mini-Jamba",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = from_pretrained("TechxGenus/Mini-Jamba")

st = time.time()
outputs, count = model.generate(tokenizer, prompt, max_tokens=188, sample=False, top_k=40)
et = time.time()

print(et-st)
print(count)
print(outputs)
