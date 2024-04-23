# https://github.com/alxndrTL/mamba.py/issues/26

import torch

import sys
sys.path.append('..')
from mamba import MambaBlock, MambaConfig

Bs, L, D, N = 2, 64, 32, 16

config = MambaConfig(d_model=D, n_layers=0, use_cuda=True)
model = MambaBlock(config).to("cuda")

# API for selective_scan() and selective_scan_seq() 
# x : (Bs, L, ED)
# Δ : (Bs, L, ED)
# A : (ED, N)
# B : (Bs, L, N)
# C : (Bs, L, N)
# D : (ED)

# y : (Bs, L, ED)

x = torch.randn(Bs, L, 2*D).to("cuda") # x.requieres_grad = True
delta = torch.randn(Bs, L, 2*D).to("cuda")
A = torch.randn(2*D, N).to("cuda")
B = torch.randn(Bs, L, N).to("cuda")
C = torch.randn(Bs, L, N).to("cuda")
D = torch.randn(2*D,).to("cuda")

y_pscan = model.selective_scan(x, delta, A, B, C, D)
y_seq = model.selective_scan_seq(x, delta, A, B, C, D)

print(y_pscan)
print(y_seq)

print(torch.allclose(y_seq, y_pscan, rtol=0.01))