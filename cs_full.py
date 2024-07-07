import torch
import torch.nn as nn

torch.manual_seed(123456)

L = 1024

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

x = torch.randn(L).to("cuda") # L elements en RAM (pas exactement L, sorte de npo2)
x.requires_grad = True

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")

fc = nn.Linear(1, 128).to("cuda")

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")

h = torch.cumsum(x, dim=0) # L elements en RAM
out = fc(h.unsqueeze(1))

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")

J = out.sum() # 128 elements en RAM

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")

J.backward() # 1024 elements

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")

print(x.grad)

"""
J.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (4)} elements")
"""