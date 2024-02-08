import torch

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 64, 128, 16

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=True, rev=True, version='2')
model = Mamba(config).to(device)

X = torch.randn(B, L, D).to(device, non_blocking=True)

model(X)

torch.cuda.memory._record_memory_history()
output = model(X)
loss = output.sum()
loss.backward()

torch.cuda.memory._dump_snapshot("v2.pickle")