import torch
import torch.autograd.profiler as profiler

#from pscan import pscan
from pscan_mem import pscan

B, L, D, N = 16, 1024, 32, 16

A = torch.randn(B, L, D, N).to("cuda")
X = torch.randn(B, L, D, N).to("cuda")

H = pscan(A, X)
