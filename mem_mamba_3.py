import torch

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 64, 128, 16

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=True, rev=False, version='2')
model = Mamba(config).to(device)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

torch.cuda.reset_peak_memory_stats(device)
initial_memory = torch.cuda.max_memory_allocated(device)

for _ in range(100):
    X = torch.randn(B, L, D).to(device, non_blocking=True)

    output = model(X)
    loss = output.sum()

    loss.backward()

peak_memory = torch.cuda.max_memory_allocated(device=device)  # Peak memory during backward

print(initial_memory/(1024**2))
print(peak_memory/(1024**2))
print("-----------------------------")
