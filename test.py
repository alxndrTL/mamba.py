import torch

from mambapy.mamba import MambaConfig, Mamba

config = MambaConfig(d_model=64, n_layers=2, pscan=False)
model = Mamba(config).to("cuda")

x = torch.randn(16, 5000, 64).to("cuda")
y = model(x)