#!/usr/bin/env python3
"""Fast Metal pscan test."""
import torch
import metal_pscan._C as _C

print("=== Test 1: Small (B=1, L=32, D=1, N=1) ===")
B, L, D, N = 1, 32, 1, 1
A = torch.ones(B, L, D, N, device='mps', dtype=torch.float32) * 0.9
X = torch.ones(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

H = _C.forward(A, X)
torch.mps.synchronize()

# Quick check: H[L-1] should be sum of geometric series
# H[n] = sum_{i=0}^{n} 0.9^(n-i) = (1 - 0.9^(n+1)) / (1 - 0.9)
expected_last = (1 - 0.9**(L)) / (1 - 0.9)
actual_last = H[0, -1, 0, 0].item()
print(f"H[last] = {actual_last:.4f}, expected = {expected_last:.4f}, error = {abs(actual_last - expected_last):.6f}")

print("\n=== Test 2: Medium (B=2, L=64, D=32, N=8) ===")
B, L, D, N = 2, 64, 32, 8
torch.manual_seed(42)
A = torch.rand(B, L, D, N, device='mps', dtype=torch.float32) * 0.4 + 0.5
X = torch.randn(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

H_metal = _C.forward(A, X)
torch.mps.synchronize()

# Use mambapy's pscan as reference (vectorized, fast)
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/mamba-metal')
from mambapy.pscan import pscan as pytorch_pscan

H_ref = pytorch_pscan(A, X)
torch.mps.synchronize()

error = (H_metal - H_ref).abs().max().item()
print(f"Max error vs PyTorch pscan: {error:.6f}")
if error < 1e-4:
    print("PASSED!")
else:
    print(f"FAILED! (error too large)")

print("\n=== Benchmark ===")
import time

B, L, D, N = 2, 256, 128, 16
A = torch.rand(B, L, D, N, device='mps', dtype=torch.float32) * 0.4 + 0.5
X = torch.randn(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

# Warmup
for _ in range(5):
    _ = _C.forward(A, X)
torch.mps.synchronize()

# Metal timing
iters = 50
torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    _ = _C.forward(A, X)
torch.mps.synchronize()
metal_time = (time.perf_counter() - t0) / iters * 1000

# PyTorch timing
for _ in range(5):
    _ = pytorch_pscan(A, X)
torch.mps.synchronize()

torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    _ = pytorch_pscan(A, X)
torch.mps.synchronize()
pytorch_time = (time.perf_counter() - t0) / iters * 1000

print(f"Shape: B={B}, L={L}, D={D}, N={N}")
print(f"Metal:   {metal_time:.3f} ms")
print(f"PyTorch: {pytorch_time:.3f} ms")
print(f"Speedup: {pytorch_time/metal_time:.2f}x")
