import torch

from mamba_lm import MambaLM, MambaLMConfig

import time

def nano_time(func, inps, *, iterations=100):
    start_time = time.time_ns()

    for _ in range(iterations):
        func(inps)

    end_time = time.time_ns()

    total_time_ns = end_time - start_time
    return total_time_ns

config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=32000)
model = MambaLM(config)

# Changing backend to inductor causes accuracy errors!
compiled_model = torch.compile(backend="eager", fullgraph=True)(model)

x = torch.randint(high=32000, size=(16, 64))
ref = model(x)
# First run (cheat by preheating, lol)
logits = compiled_model(x) # (B, L, vocab_size)
assert torch.equal(logits, ref)

eager_time = nano_time(model, x)
print("Eager time:", eager_time)
compile_time = nano_time(compiled_model, x)
print("Compiled time:", compile_time)

print("Speedup or slowdown?", eager_time / compile_time)
