import math
from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import mlx.nn as nn

from pscan_mlx import pscan
from misc import softplus, unsqueeze, clamp, DepthWiseConv1d


"""

This file closely follows the mamba.py written in PyTorch.
The torch->mlx conversion is pretty straightforward, instead for one particular (and temporary) point : depthwise convolution.
As of release v0.0.10, mlx doesn't supports "groups" other than 1 for 1d convolutions. But in the official implementation, a depthwise 1d conv is used (ie, set groups to the number of channels).
A workaround is to actually do a convolution with groups=1, but zero out all the elements of the conv weights except those on the "diagonal". (see misc.py)

A Mamba model is composed of several layers, which are ResidualBlock.
A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""

# TODO use cpu for some ops ? investigate

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = False # use parallel scan mode or sequential mode when training. on MLX, the pscan isn't performant.

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = [ResidualBlock(config) for _ in range(config.n_layers)]
        #self.norm_f = RMSNorm(config.d_model)

    def __call__(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        #x = self.norm_f(x)

        return x

    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = nn.RMSNorm(config.d_model)

    def __call__(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, d_conv-1, ED)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # short 1d conv over time
        self.conv1d = DepthWiseConv1d(channels=config.d_inner, kernel_size=config.d_conv,
                                      bias=config.conv_bias, padding=config.d_conv-1)

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
 
        # TODO: disable grad ?
        # see https://pytorch.org/docs/stable/nn.init.html
        if config.dt_init == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif config.dt_init == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = clamp(mx.exp(
            mx.random.uniform(shape=[config.d_inner]) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ), min=config.dt_init_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_proj.bias = inv_dt

        # S4D real initialization
        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=config.d_inner, axis=0)
        self.A_log = mx.log(A) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = mx.ones([config.d_inner])

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def __call__(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.split(indices_or_sections=2, axis=2) # (B, L, ED), (B, L, ED)

        # x branch
        x = self.conv1d(x)[:, :L, :]

        x = nn.silu(x)
        y = self.ssm(x)

        # z branch
        z = nn.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -mx.exp(self.A_log) # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.config.dt_rank, self.config.dt_rank+self.config.d_state], axis=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        
        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        h = mx.zeros([x.shape[0], self.config.d_inner, self.config.d_state]) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = mx.stack(hs, axis=1)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        
        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, d_conv-1, ED)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.split(indices_or_sections=2, axis=1) # (B, ED), (B, ED)

        # x branch
        x_cache = unsqueeze(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.config.d_conv-1, :] # (B, ED)

        x = nn.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = nn.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1) # (B, d_conv-1, ED)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -mx.exp(self.A_log) # (ED, N) # todo : move out of step (timestep independent)
        D = self.D

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.config.dt_rank, self.config.dt_rank+self.config.d_state], axis=-1) # (B, dt_rank), (B, N), (B, N)
        delta = softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1) # (B, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, ED, N)

        if h is None:
            h = mx.zeros([x.shape[0], self.config.d_inner, self.config.d_state]) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ unsqueeze(C, -1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x
        
        return y, h
