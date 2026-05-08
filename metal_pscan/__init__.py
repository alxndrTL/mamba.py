"""
Metal PScan - Native Metal parallel scan for Mamba on Apple Silicon

Usage:
    from metal_pscan import metal_pscan, is_available

    if is_available():
        H = metal_pscan(A, X)  # Drop-in replacement for PyTorch pscan
"""

import torch
from typing import Tuple

# Try to import the C++ extension
_HAS_METAL = False
_C = None

try:
    # The extension is built as metal_pscan._C
    import metal_pscan._C as _C
    _HAS_METAL = _C.is_available()

    # Register torch.compile ops (import triggers registration)
    try:
        from . import torch_ops
    except Exception as e:
        pass  # torch_ops is optional, don't fail if it can't load

except ImportError as e:
    print(f"Metal pscan C++ extension not found: {e}")
    print("Build with: cd mamba-metal && python setup_metal.py build_ext --inplace")
    _HAS_METAL = False

def is_available() -> bool:
    """Check if Metal pscan is available."""
    return _HAS_METAL

class MetalPScan(torch.autograd.Function):
    """PyTorch autograd function for Metal parallel scan."""

    @staticmethod
    def forward(ctx, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of parallel scan.

        Args:
            A: (B, L, D, N) decay coefficients
            X: (B, L, D, N) input values

        Returns:
            H: (B, L, D, N) accumulated hidden states where H[t] = A[t] * H[t-1] + X[t]
        """
        # Run Metal kernel
        H = _C.forward(A, X)

        # Save for backward
        ctx.save_for_backward(A, X, H)

        return H

    @staticmethod
    def backward(ctx, grad_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward pass."""
        A, X, H = ctx.saved_tensors

        # Run Metal backward kernel
        grad_A, grad_X = _C.backward(A, X, H, grad_H)

        return grad_A, grad_X

def metal_pscan(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Metal-accelerated parallel scan for Mamba.

    Drop-in replacement for mambapy.pscan.pscan

    Args:
        A: (B, L, D, N) decay coefficients
        X: (B, L, D, N) input values

    Returns:
        H: (B, L, D, N) where H[t] = A[t] * H[t-1] + X[t]
    """
    if not _HAS_METAL:
        raise RuntimeError(
            "Metal pscan not available. "
            "Build with: cd mamba-metal && python setup_metal.py build_ext --inplace"
        )

    return MetalPScan.apply(A, X)

# Alias for compatibility
pscan = metal_pscan
