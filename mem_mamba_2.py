import torch

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 64, 128, 16

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=True, rev=False, version='2')
model = Mamba(config).to(device)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

initial_memory = torch.cuda.max_memory_allocated(device)

X = torch.randn(B, L, D).to(device, non_blocking=True)

output = model(X)
loss = output.sum()

peak_memory_during_forward = torch.cuda.max_memory_allocated(device)

loss.backward()

peak_memory_during_backward = torch.cuda.max_memory_allocated(device=device)  # Peak memory during backward

forward_memory_mb = (peak_memory_during_forward - initial_memory) / (1024 ** 2)
backward_memory_mb = (peak_memory_during_backward - peak_memory_during_forward) / (1024 ** 2)
total_peak_memory_mb = peak_memory_during_backward / (1024 ** 2)

# Print the memory consumption in MB

print(f"Initial memory: {initial_memory / (1024 ** 2):.2f} MB")
print(f"Forward pass peak memory: {forward_memory_mb:.2f} MB")
print(f"Backward pass additional peak memory: {backward_memory_mb:.2f} MB")
print(f"Total peak memory: {total_peak_memory_mb:.2f} MB")
