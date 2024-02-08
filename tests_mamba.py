import timeit
import torch

from mamba import Mamba, MambaConfig

device = "cuda"

B, L, D, N = 16, 64, 128, 16

def train(model, num_steps):
    for _ in range(num_steps):
        X = torch.randn(B, L, D).to(device, non_blocking=True)
        Y = model(X).sum()
        Y.backward()

##### UNFOLDING
print("with unfolding")

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=True)
model = Mamba(config).to(device)

# warm ups
for _ in range(5):
    train(model, num_steps=10)
    
time2 = timeit.timeit(lambda: train(model, num_steps=10), number=5)

########### NO UNFOLDING
print("with no unfolding")

config = MambaConfig(d_model=D, n_layers=8, d_state=N, unfolded=False)
model = Mamba(config).to(device)

# warm ups
for _ in range(5):
    train(model, num_steps=10)

time1 = timeit.timeit(lambda: train(model, num_steps=10), number=5)

print(time1/100)
print(time2/100)
