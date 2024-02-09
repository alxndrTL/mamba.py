import torch
import torch.autograd.profiler as profiler

from pscan import pscan
from pscan_unfolded import pscan as pscan_unfolded
from pscan_unfolded_rev import pscan as pscan_unfolded_rev

B, L, D, N = 16, 1024, 32, 16

A = torch.randn(B, L, D, N).to("cuda")
X = torch.randn(B, L, D, N).to("cuda")

H = pscan_unfolded_rev(A, X)
with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("pscan_custom_function"):
        H = pscan_unfolded_rev(A, X)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
prof.export_chrome_trace("pscan_profiling_trace.json")

print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

