import math

import torch
import torch.nn.functional as F
import torch.autograd.profiler as profiler

B, L, D, N = 16, 1024, 128, 16

X = torch.randn(B, L, D, N).to("cuda")

X.clone()
with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("pscan_custom_function"):
        X.clone()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

######################################

def pad_npo2(X):
    # X : (B, L, D, N)

    # Y : (B, npo2(L), D, N)

    len_npo2 = 2**math.ceil(math.log2(X.size(1)))
    pad_tuple = (0, 0, 0, 0, len_npo2 - X.size(1), 0)
    return F.pad(X, pad_tuple, "constant", 0)

pad_npo2(X)
with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("pscan_custom_function"):
        pad_npo2(X)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")

######################################

def pad_npo2_custom(X):
    # X : (B, L, D, N)

    # Y : (B, npo2(L), D, N)

    len_npo2 = 2**math.ceil(math.log2(X.size(1)))
    zeros = torch.zeros(B, len_npo2, D, N, device="cuda")
    zeros[:X.size(1)] = X
    return zeros

pad_npo2_custom(X)
with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
    with profiler.record_function("pscan_custom_function"):
        pad_npo2_custom(X)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
print(f"Peak CUDA Memory Usage: {prof.total_average().cuda_memory_usage / (1024 ** 2)} MB")