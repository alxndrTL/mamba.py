import torch
import torch.nn as nn

from mem_utils import log_mem

#from pscan import pscan
from pscan_mem import pscan

class PScanModel(nn.Module):
    def __init__(self, ):
        super().__init__() 

    def forward(self, A, X):
        return pscan(A, X)

device = "cuda"

B, L, D, N = 1, 512, 32, 16

model = PScanModel()

A = torch.randn(B, L, D, N, requires_grad=True).to("cuda")
X = torch.randn(B, L, D, N, requires_grad=True).to("cuda")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
initial_memory = torch.cuda.max_memory_allocated(device)

Y = pscan(A, X)

J = Y.sum()
J.backward()

peak_memory = torch.cuda.max_memory_allocated(device=device)

print(initial_memory/(1024**2))
print(peak_memory/(1024**2))
print("-----------------------------")
