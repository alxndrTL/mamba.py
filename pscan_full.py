import torch

from mambapy.pscan import pscan

B, L, D, N = 100, 1024, 64, 16

# full pscan

torch.cuda.reset_peak_memory_stats(device=None)

torch.manual_seed(123456)
A = torch.ones(B, L, D, N, requires_grad=True, device="cuda")
X = torch.randn(B, L, D, N, requires_grad=True, device="cuda")

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

hs = pscan(A, X)

J = hs.sum()
J.backward()

"""
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

hs = pscan(A, X)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

J = hs.sum()
J.backward()
"""

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")


print(J)
print(A.grad.mean())
print(X.grad.mean())
