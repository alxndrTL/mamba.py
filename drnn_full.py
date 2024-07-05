import torch
import torch.nn as nn
import torch.optim as optim

from rnn import RNNConfig, RNN

# 1000 = OOM
B, L, D = 512, 1024, 64

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

config = RNNConfig(d_model=D, n_layers=2)
model = RNN(config).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(B, L, D).to("cuda")
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB") # torch.cuda.max_memory_allocated(device=None)

# classic forward
hidden = None
optimizer.zero_grad()

output = model(x, [torch.zeros(B, D, device="cuda"), torch.zeros(B, D, device="cuda")])
loss = output.sum()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

loss.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

print("Done")

print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")