#!/usr/bin/env python3
"""Minimal test - just check if forward runs at all."""
import torch
print("Import torch OK")

import metal_pscan._C as _C
print(f"Import _C OK, available: {_C.is_available()}")

# Tiny test
B, L, D, N = 1, 32, 1, 1
A = torch.ones(B, L, D, N, device='mps', dtype=torch.float32) * 0.9
X = torch.ones(B, L, D, N, device='mps', dtype=torch.float32)
print(f"Created tensors: A={A.shape}, X={X.shape}")

print("Calling _C.forward...")
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Metal kernel timed out!")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    H = _C.forward(A, X)
    signal.alarm(0)
    print(f"Forward returned: H={H.shape}")
    print(f"H values: {H.flatten().tolist()}")
except TimeoutError as e:
    print(f"TIMEOUT: {e}")
except Exception as e:
    print(f"ERROR: {e}")
