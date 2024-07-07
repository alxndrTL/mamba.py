import torch

from mambapy.mamba import MambaConfig, Mamba

B, L, D = 42, 240, 16

config = MambaConfig(d_model=D, n_layers=3)
model = Mamba(config).to("cuda")

x = torch.randn(B, L, D, device="cuda")
x.requires_grad = True

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

output, _ = model(x)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

J = output.sum()
J.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")