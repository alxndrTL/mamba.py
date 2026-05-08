#!/usr/bin/env python3
"""Benchmark Metal pscan scaling."""
import torch
import time
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/mamba-metal')

import metal_pscan._C as _C
from mambapy.pscan import pscan as pytorch_pscan

print("=" * 60)
print("Metal PScan vs PyTorch PScan on MPS")
print("=" * 60)

B, D, N = 2, 128, 16
print(f"\nConfig: B={B}, D={D}, N={N}")
print(f"\n{'Seq Len':>8} | {'Metal (ms)':>12} | {'PyTorch (ms)':>13} | {'Speedup':>8}")
print("-" * 52)

for L in [64, 128, 256, 512, 1024]:
    A = torch.rand(B, L, D, N, device='mps', dtype=torch.float32) * 0.4 + 0.5
    X = torch.randn(B, L, D, N, device='mps', dtype=torch.float32)
    torch.mps.synchronize()

    # Warmup
    for _ in range(10):
        _ = _C.forward(A, X)
        _ = pytorch_pscan(A, X)
    torch.mps.synchronize()

    iters = 100

    # Metal
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _C.forward(A, X)
    torch.mps.synchronize()
    metal_ms = (time.perf_counter() - t0) / iters * 1000

    # PyTorch
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pytorch_pscan(A, X)
    torch.mps.synchronize()
    pytorch_ms = (time.perf_counter() - t0) / iters * 1000

    speedup = pytorch_ms / metal_ms
    print(f"{L:>8} | {metal_ms:>12.3f} | {pytorch_ms:>13.3f} | {speedup:>7.2f}x")

print()
