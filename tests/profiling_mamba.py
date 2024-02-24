import torch
import torch.autograd.profiler as profiler

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 1024, 1024, 16

config = MambaConfig(d_model=D, n_layers=8, d_state=N)
model = Mamba(config).to(device)

X = torch.randn(B, L, D).to(device, non_blocking=True)

model(X)

with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("model_forward"):
        output = model(X)
        loss = output.sum()
    with profiler.record_function("model_backward"):
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")
