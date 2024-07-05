import torch

from mambapy.pscan import pscan

B, L, D, N = 16, 256, 58, 6

# full pscan

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

torch.manual_seed(123456)
A = torch.randn(B, L, D, N, requires_grad=True, device="cuda")
X = torch.randn(B, L, D, N, requires_grad=True, device="cuda")

hs = pscan(A, X)

J = hs.sum()
J.backward()

print(f"gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
