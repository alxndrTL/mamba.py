import torch

from mambapy.mamba import MambaConfig, Mamba

B, L, D = 42, 240, 16

config = MambaConfig(d_model=D, n_layers=3)
model = Mamba(config).to("cuda")

x = torch.randn(B, L, D, device="cuda")
x.requires_grad = True

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

hiddens = [torch.zeros(B, 2*D, 16, device="cuda") for _ in range(config.n_layers)]

chunk = x[:, 0:120]
output, hiddens = model(chunk, hiddens)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

J = output.sum()
J.backward(retain_graph=True)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

chunk = x[:, 120:]
output, _ = model(chunk, hiddens)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

J = output.sum()
J.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")