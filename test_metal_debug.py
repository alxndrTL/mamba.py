#!/usr/bin/env python3
"""Test Metal pscan correctness and performance."""
import torch
import sys
sys.path.insert(0, '/Users/zimski/projects/oss/mamba-metal')

import metal_pscan._C as _C
from mambapy.pscan import pscan as pytorch_pscan

print("=== Test 1: Tiny (B=1, L=4, D=1, N=1) ===")
A = torch.tensor([[[[0.9]], [[0.9]], [[0.9]], [[0.9]]]], device='mps', dtype=torch.float32)
X = torch.tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]]]], device='mps', dtype=torch.float32)
torch.mps.synchronize()

H = _C.forward(A, X)
torch.mps.synchronize()

expected = [1.0, 1.9, 2.71, 3.439]
actual = H.flatten().tolist()
print(f"Expected: {expected}")
print(f"Actual:   {[round(x, 3) for x in actual]}")
print(f"PASSED!" if all(abs(a - e) < 0.01 for a, e in zip(actual, expected)) else "FAILED!")

print("\n=== Test 2: Small (B=1, L=32, D=1, N=1) ===")
B, L, D, N = 1, 32, 1, 1
A = torch.ones(B, L, D, N, device='mps', dtype=torch.float32) * 0.9
X = torch.ones(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

H = _C.forward(A, X)
torch.mps.synchronize()

# H[L-1] = geometric series sum
expected_last = (1 - 0.9**L) / (1 - 0.9)
actual_last = H[0, -1, 0, 0].item()
error = abs(actual_last - expected_last)
print(f"H[last] = {actual_last:.4f}, expected = {expected_last:.4f}, error = {error:.6f}")
print(f"PASSED!" if error < 1e-4 else "FAILED!")

print("\n=== Test 3: Medium (B=2, L=64, D=32, N=8) ===")
B, L, D, N = 2, 64, 32, 8
torch.manual_seed(42)
A = torch.rand(B, L, D, N, device='mps', dtype=torch.float32) * 0.4 + 0.5
X = torch.randn(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

H_metal = _C.forward(A, X)
H_ref = pytorch_pscan(A, X)
torch.mps.synchronize()

error = (H_metal - H_ref).abs().max().item()
print(f"Max error vs PyTorch pscan: {error:.6f}")
print(f"PASSED!" if error < 1e-4 else "FAILED!")

print("\n=== Test 4: Large (B=2, L=1024, D=128, N=16) ===")
B, L, D, N = 2, 1024, 128, 16
torch.manual_seed(123)
A = torch.rand(B, L, D, N, device='mps', dtype=torch.float32) * 0.4 + 0.5
X = torch.randn(B, L, D, N, device='mps', dtype=torch.float32)
torch.mps.synchronize()

H_metal = _C.forward(A, X)
H_ref = pytorch_pscan(A, X)
torch.mps.synchronize()

error = (H_metal - H_ref).abs().max().item()
print(f"Max error vs PyTorch pscan: {error:.6f}")
print(f"PASSED!" if error < 1e-3 else "FAILED!")

print("\n=== All tests completed ===")
