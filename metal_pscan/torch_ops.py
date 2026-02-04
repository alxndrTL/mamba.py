"""
torch.compile support for metal-pscan custom ops.

This registers the Metal ops with torch.library so that torch.compile
can trace through them instead of falling back to eager mode.

Usage:
    import metal_pscan.torch_ops  # Register ops on import

Then use torch.ops.metal_pscan.* instead of _C.* directly.
"""

import torch
import torch.nn.functional as F
from torch.library import Library, impl

# Use register_fake (new name) or impl_abstract (old name) for compatibility
try:
    from torch.library import register_fake
except ImportError:
    from torch.library import impl_abstract as register_fake

# Create library namespace
pscan_lib = Library("metal_pscan", "DEF")

# =============================================================================
# Op Definitions
# =============================================================================

# conv1d_silu: Fused depthwise conv1d + SiLU activation
# Input: x (B, D, L), weight (D, 1, d_conv), bias (D)
# Output: (B, D, L)
pscan_lib.define(
    "conv1d_silu(Tensor x, Tensor weight, Tensor bias) -> Tensor"
)

# ssm_output_fused: Super-fused SSM (prep + pscan + output matmul)
# Computes y[b,l,d] = sum_n(h[b,l,d,n] * C[b,l,n]) + D[d] * x[b,l,d]
pscan_lib.define(
    "ssm_output_fused(Tensor delta, Tensor A, Tensor B_ssm, Tensor x, Tensor C_ssm, Tensor D_param) -> Tensor"
)

# Forward pass: parallel scan
pscan_lib.define(
    "forward(Tensor A, Tensor X) -> Tensor"
)

# Backward pass: parallel scan gradients
pscan_lib.define(
    "backward(Tensor A, Tensor X, Tensor H, Tensor grad_H) -> (Tensor, Tensor)"
)

# ssm_fused: Fused SSM (prep + pscan, returns hidden states)
pscan_lib.define(
    "ssm_fused(Tensor delta, Tensor A, Tensor B_ssm, Tensor x) -> Tensor"
)

# =============================================================================
# MPS Implementations (call the C++ kernels)
# =============================================================================

@impl(pscan_lib, "conv1d_silu", "MPS")
def conv1d_silu_mps(x, weight, bias):
    from . import _C
    return _C.conv1d_silu(x, weight, bias)


@impl(pscan_lib, "ssm_output_fused", "MPS")
def ssm_output_fused_mps(delta, A, B_ssm, x, C_ssm, D_param):
    from . import _C
    return _C.ssm_output_fused(delta, A, B_ssm, x, C_ssm, D_param)


@impl(pscan_lib, "forward", "MPS")
def forward_mps(A, X):
    from . import _C
    return _C.forward(A, X)


@impl(pscan_lib, "backward", "MPS")
def backward_mps(A, X, H, grad_H):
    from . import _C
    return _C.backward(A, X, H, grad_H)


@impl(pscan_lib, "ssm_fused", "MPS")
def ssm_fused_mps(delta, A, B_ssm, x):
    from . import _C
    return _C.ssm_fused(delta, A, B_ssm, x)


# =============================================================================
# Meta/Fake Implementations (for tracing - shapes only, no compute)
# =============================================================================

@register_fake("metal_pscan::conv1d_silu")
def conv1d_silu_meta(x, weight, bias):
    # x: (B, D, L) -> output: (B, D, L) same shape
    return x.new_empty(x.shape)


@register_fake("metal_pscan::ssm_output_fused")
def ssm_output_fused_meta(delta, A, B_ssm, x, C_ssm, D_param):
    # delta: (B, L, D) -> output: (B, L, D) same shape
    return delta.new_empty(delta.shape)


@register_fake("metal_pscan::forward")
def forward_meta(A, X):
    # A, X: (B, L, D, N) -> H: (B, L, D, N) same shape
    return X.new_empty(X.shape)


@register_fake("metal_pscan::backward")
def backward_meta(A, X, H, grad_H):
    # Returns grad_A, grad_X same shapes as A, X
    return A.new_empty(A.shape), X.new_empty(X.shape)


@register_fake("metal_pscan::ssm_fused")
def ssm_fused_meta(delta, A, B_ssm, x):
    # delta: (B, L, D), A: (D, N) -> H: (B, L, D, N)
    B, L, D = delta.shape
    N = A.shape[1]
    return delta.new_empty(B, L, D, N)


# =============================================================================
# Autograd Functions with manual backward
# =============================================================================

class Conv1dSiluFunction(torch.autograd.Function):
    """Autograd wrapper for fused conv1d + silu with manual backward."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        # x: (B, D, L), weight: (D, 1, d_conv), bias: (D)
        from . import _C

        # Save for backward
        ctx.save_for_backward(x, weight, bias)

        # Call Metal kernel
        return _C.conv1d_silu(x.contiguous(), weight.contiguous(), bias.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        B, D, L = x.shape
        d_conv = weight.shape[2]

        # Store original dtype for output
        out_dtype = x.dtype

        # CRITICAL: Do backward in float32 to avoid NaN from float16 overflow
        compute_dtype = torch.float32
        x = x.to(compute_dtype)
        weight = weight.to(compute_dtype)
        bias = bias.to(compute_dtype)
        grad_output = grad_output.to(compute_dtype)

        # Forward recompute for silu backward (need pre-activation values)
        # conv1d output before silu
        conv_out = F.conv1d(x, weight, bias, groups=D, padding=d_conv - 1)[:, :, :L]

        # silu backward: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #                                     = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid_conv = torch.sigmoid(conv_out)
        silu_grad = sigmoid_conv * (1.0 + conv_out * (1.0 - sigmoid_conv))

        # grad w.r.t. conv output
        grad_conv = grad_output * silu_grad

        # conv1d backward for depthwise convolution
        # Original: conv1d with padding=d_conv-1, then slice [:L] (causal conv)
        # grad_x: use conv_transpose1d with output_padding to get correct size
        grad_x = F.conv_transpose1d(
            grad_conv, weight,
            groups=D,
            padding=d_conv - 1,
            output_padding=d_conv - 1
        )[:, :, :L]

        # grad_weight: for depthwise conv, correlate input with grad_conv
        # weight shape: (D, 1, d_conv), we need grad for each channel independently
        # Use conv1d with input as "kernel" - but simpler to just do it directly
        grad_weight = torch.zeros_like(weight)
        x_padded = F.pad(x, (d_conv - 1, 0))  # left pad for causal conv
        for k in range(d_conv):
            grad_weight[:, 0, k] = (x_padded[:, :, k:k+L] * grad_conv).sum(dim=(0, 2))

        # grad_bias: sum over batch and length
        grad_bias = grad_conv.sum(dim=(0, 2))

        # Convert back to original dtype
        return grad_x.to(out_dtype), grad_weight.to(out_dtype), grad_bias.to(out_dtype)


class SSMOutputFusedFunction(torch.autograd.Function):
    """Autograd wrapper for fused SSM output with manual backward."""

    @staticmethod
    def forward(ctx, delta, A, B_ssm, x, C_ssm, D_param):
        # delta: (B, L, D), A: (D, N), B_ssm: (B, L, N), x: (B, L, D), C_ssm: (B, L, N), D_param: (D)
        from . import _C

        # Save for backward
        ctx.save_for_backward(delta, A, B_ssm, x, C_ssm, D_param)

        # Call Metal kernel
        return _C.ssm_output_fused(
            delta.contiguous(), A.contiguous(), B_ssm.contiguous(),
            x.contiguous(), C_ssm.contiguous(), D_param.contiguous()
        )

    @staticmethod
    def backward(ctx, grad_y):
        delta, A, B_ssm, x, C_ssm, D_param = ctx.saved_tensors
        B, L, D = delta.shape
        N = A.shape[1]

        # CRITICAL: Do backward in float32 to avoid NaN from float16 overflow
        # The forward Metal kernel handles float16 fine, but Python backward needs float32
        compute_dtype = torch.float32
        delta = delta.to(compute_dtype)
        A = A.to(compute_dtype)
        B_ssm = B_ssm.to(compute_dtype)
        x = x.to(compute_dtype)
        C_ssm = C_ssm.to(compute_dtype)
        D_param = D_param.to(compute_dtype)
        grad_y = grad_y.to(compute_dtype)

        # Store original dtype for output
        out_dtype = ctx.saved_tensors[0].dtype

        # Recompute forward for gradient computation
        # deltaA = exp(delta * A), shape: (B, L, D, N)
        # Clamp the exponent to avoid overflow (A is negative, delta positive, so product is negative)
        delta_A_product = (delta.unsqueeze(-1) * A).clamp(-20, 20)  # exp(-20) ≈ 2e-9, exp(20) ≈ 5e8
        deltaA = torch.exp(delta_A_product)

        # BX = delta * B * x, shape: (B, L, D, N)
        # delta: (B,L,D), B_ssm: (B,L,N), x: (B,L,D)
        BX = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)

        # Sequential scan to get hidden states h (can't avoid this for backward)
        # h[l] = deltaA[l] * h[l-1] + BX[l]
        h = torch.zeros(B, L, D, N, device=delta.device, dtype=compute_dtype)
        h_prev = torch.zeros(B, D, N, device=delta.device, dtype=compute_dtype)
        for l in range(L):
            h_prev = deltaA[:, l] * h_prev + BX[:, l]
            h[:, l] = h_prev

        # y = sum_n(h * C) + D * x
        # grad_y: (B, L, D)

        # grad_D_param = sum over B,L of grad_y * x
        grad_D_param = (grad_y * x).sum(dim=(0, 1))

        # grad_x from D*x term
        grad_x_D = grad_y * D_param

        # grad from h*C term - need reverse scan for gradients
        # grad_h: (B, L, D, N) from grad_y * C
        grad_h = grad_y.unsqueeze(-1) * C_ssm.unsqueeze(2)  # (B,L,D,1) * (B,L,1,N) -> (B,L,D,N)

        # grad_C = sum_d(grad_y * h)
        grad_C = (grad_y.unsqueeze(-1) * h).sum(dim=2)  # (B, L, N)

        # Reverse scan for gradients through the recurrence
        # h[l] = deltaA[l] * h[l-1] + BX[l]
        # dL/dh[l-1] = dL/dh[l] * deltaA[l]
        # dL/ddeltaA[l] = dL/dh[l] * h[l-1]
        # dL/dBX[l] = dL/dh[l]

        grad_deltaA = torch.zeros_like(deltaA)
        grad_BX = torch.zeros_like(BX)

        grad_h_acc = torch.zeros(B, D, N, device=delta.device, dtype=compute_dtype)

        # Collect h_prev values for each timestep (h[l-1] needed for grad of h[l])
        # h_prev_for_l[l] = h[l-1], where h[-1] = 0
        h_prev_for_l = torch.zeros(B, L, D, N, device=delta.device, dtype=compute_dtype)
        h_prev = torch.zeros(B, D, N, device=delta.device, dtype=compute_dtype)
        for l in range(L):
            h_prev_for_l[:, l] = h_prev
            h_prev = deltaA[:, l] * h_prev + BX[:, l]

        # Backward pass - clamp grad_h_acc to prevent explosion
        grad_clip = 1e4  # max gradient magnitude
        for l in range(L - 1, -1, -1):
            grad_h_acc = grad_h_acc + grad_h[:, l]
            grad_h_acc = grad_h_acc.clamp(-grad_clip, grad_clip)  # prevent explosion
            grad_BX[:, l] = grad_h_acc
            grad_deltaA[:, l] = grad_h_acc * h_prev_for_l[:, l]
            grad_h_acc = grad_h_acc * deltaA[:, l]

        # Clamp intermediate gradients to prevent NaN propagation
        grad_deltaA = grad_deltaA.clamp(-grad_clip, grad_clip)
        grad_BX = grad_BX.clamp(-grad_clip, grad_clip)

        # grad_delta from deltaA = exp(delta * A)
        # d(deltaA)/d(delta) = deltaA * A
        grad_delta = (grad_deltaA * deltaA * A).sum(dim=-1)  # sum over N

        # grad_delta from BX = delta * B * x
        grad_delta = grad_delta + (grad_BX * B_ssm.unsqueeze(2) * x.unsqueeze(-1)).sum(dim=-1)

        # grad_A from deltaA = exp(delta * A)
        grad_A = (grad_deltaA * deltaA * delta.unsqueeze(-1)).sum(dim=(0, 1))  # sum over B, L

        # grad_B from BX = delta * B * x
        grad_B = (grad_BX * delta.unsqueeze(-1) * x.unsqueeze(-1)).sum(dim=2)  # sum over D -> (B, L, N)

        # grad_x from BX = delta * B * x
        grad_x_BX = (grad_BX * delta.unsqueeze(-1) * B_ssm.unsqueeze(2)).sum(dim=-1)  # sum over N

        grad_x = grad_x_D + grad_x_BX

        # Clamp final gradients to prevent inf/nan (float16 range is ~65504)
        max_grad = 65000.0
        grad_delta = grad_delta.clamp(-max_grad, max_grad)
        grad_A = grad_A.clamp(-max_grad, max_grad)
        grad_B = grad_B.clamp(-max_grad, max_grad)
        grad_x = grad_x.clamp(-max_grad, max_grad)
        grad_C = grad_C.clamp(-max_grad, max_grad)
        grad_D_param = grad_D_param.clamp(-max_grad, max_grad)

        # Convert back to original dtype
        grad_delta = grad_delta.to(out_dtype)
        grad_A = grad_A.to(out_dtype)
        grad_B = grad_B.to(out_dtype)
        grad_x = grad_x.to(out_dtype)
        grad_C = grad_C.to(out_dtype)
        grad_D_param = grad_D_param.to(out_dtype)

        return grad_delta, grad_A, grad_B, grad_x, grad_C, grad_D_param


# =============================================================================
# Wrapper functions that use autograd Functions
# =============================================================================

def conv1d_silu_autograd(x, weight, bias):
    """Conv1d + SiLU with autograd support."""
    return Conv1dSiluFunction.apply(x, weight, bias)


def ssm_output_fused_autograd(delta, A, B_ssm, x, C_ssm, D_param):
    """SSM output fused with autograd support."""
    return SSMOutputFusedFunction.apply(delta, A, B_ssm, x, C_ssm, D_param)


# =============================================================================
# Autograd dispatch registration
# =============================================================================

# Register fallthrough for Autograd - tells PyTorch these ops support autograd
# via the standard mechanism (tracing through forward)
pscan_lib.impl("conv1d_silu", torch.library.fallthrough_kernel, "Autograd")
pscan_lib.impl("ssm_output_fused", torch.library.fallthrough_kernel, "Autograd")


print("✓ metal_pscan torch.compile ops registered (autograd fallthrough)")
