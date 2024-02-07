import timeit

import torch

from pscan import pscan
from pscan_unfolded import pscan as pscan_unfolded

B, L, D, N = 16, 512, 16, 16

A = torch.randn(B, L, D, N).to("cuda")
X = torch.randn(B, L, D, N).to("cuda")

# warm ups
for _ in range(5):
    pscan(A, X)

time1 = timeit.timeit(lambda: pscan(A, X), number=100)
time2 = timeit.timeit(lambda: pscan_unfolded(A, X), number=100)

print(time1)
print(time2)

