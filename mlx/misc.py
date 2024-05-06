import mlx.core as mx
import mlx.nn as nn

import torch

"""

This is a temporary file, as it contains additional functions which are needed but not yet implemented in MLX (as of release v0.0.10).
The first functions are straightforward, while the depthwise 1d convolution is a bit more elaborared.

"""

def softplus(x, beta=1, threshold=20):
    scaled_x = beta * x
    mask = scaled_x > threshold
    return mx.where(mask, x, 1/beta * mx.logaddexp(0, x))

def unsqueeze(x, axis):
    """
    Same API as PyTorch.
    """

    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)

def clamp(x, min=None, max=None):
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max

    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)
    
    return mx.where(mask_upper, max, x)

def topk(x, k):
    """
    Returns the top k biggest values of x along the 2nd dim.

    Args:
        x : (B, vocab_size). can be probs or logits

    Returns:
        values : (B, k). ordered from lowest to biggest val
    """

    return mx.sort(x)[:, -k:]

class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding
        
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, bias=True, padding=padding)

        # see comment below
        indices = mx.arange(channels)
        mask = mx.zeros_like(self.conv1d.weight)
        mask[indices, :, indices] = 1
        self.conv1d.weight *= mask
    
    def __call__(self, x):
        return self.conv1d(x)
    
def torch_to_mlx_depthwise_weights(torch_weights):
    """
    
    A convolution is said to be "depthwise" when channel i of the output is only computed by passing the filter overing channel i of the input.
    In torch, this is done by setting groups=number of channels.
    Because it is not yet implemented in MLX, a workaround is to zero out the weights of a conv object initialized with groups=1 (groups=1 is when output channel i is computing by passing the filter over all input channels)
    To do that, we need to zero out all elements except those on the "diagonal":
    for channels=8 and kernel_size=4, the weights are (8, 4, 8).
    these are composed of 8 x (8, 4, 1) filter, each of those is used to compute one output channel.
    this (8, 4, 1) filter is composed of 8 x (1, 4, 1) filter, each of those is passed over each input channel.
    so we need to set to 0 all those 8 filters, except the one which corresponds to the output channel of these 8 filters (so that the channels don't mix)

    """

    # torch_weights : (channels, 1, kernel_size) = (ED, 1, d_conv)

    # mlx_weights : (channels, kernel_size, channels) = (ED, d_conv, ED)

    torch_weights = torch_weights.transpose(2, 1) # (channels, kernel_size, 1) = (ED, d_conv, 1)
    channels, kernel_size, _ = torch_weights.shape

    mlx_weights = torch.zeros(channels, kernel_size, channels)

    indices = torch.arange(channels)
    if torch_weights[:, :, 0].type() == 'torch.BFloat16Tensor':
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0].float()
    else:
        mlx_weights[indices, :, indices] = torch_weights[:, :, 0]

    return mlx_weights
