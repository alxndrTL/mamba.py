import timeit
import torch

from pscan import pscan
from pscan_unfolded import pscan as pscan_unfolded
from pscan_unfolded_rev import pscan as pscan_unfolded_rev

B, L, D, N = 16, 1024, 32, 16

A = torch.randn(B, L, D, N).to("cuda")
X = torch.randn(B, L, D, N).to("cuda")

# warm ups
for _ in range(5):
    pscan(A, X)

time1 = timeit.timeit(lambda: pscan(A, X), number=1000)

# warm ups
for _ in range(5):
    pscan_unfolded(A, X)

time2 = timeit.timeit(lambda: pscan_unfolded(A, X), number=1000)

# warm ups
for _ in range(5):
    pscan_unfolded(A, X)

time3 = timeit.timeit(lambda: pscan_unfolded_rev(A, X), number=1000)

print(time1)
print(time2)
print(time3)

