import timeit
import torch

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 1024, 128, 16

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=True)
model = Mamba(config).to(device)

X = torch.randn(B, L, D).to(device)

# warm ups
for _ in range(5):
    model(X)

time2 = timeit.timeit(lambda: model(X), number=100)


###########

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=False)
model = Mamba(config).to(device)

X = torch.randn(B, L, D).to(device)

# warm ups
for _ in range(5):
    model(X)

time1 = timeit.timeit(lambda: model(X), number=100)

print(time1/100)
print(time2/100)
