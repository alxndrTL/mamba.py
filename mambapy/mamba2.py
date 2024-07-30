# Copyright (c) 2024, Tri Dao, Albert Gu.

"""

adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
It justs implements a config similar to what's being done in mamba.py.

"""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

@dataclass
class Mamba2Config:
    d_model: int # D
    n_layers: int
    d_head: int # todo : plutot n_heads non ?
    d_state: int = 64 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4
    n_groups: int = 1# todo : ??
    
    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    conv_init = None

    learnable_init_states: bool = False
    activation: str = "swish" # "swish" or "silu"
    
    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True

    chunk_size: int = 256
    use_mem_eff_path: bool = True
    dtype=None
    device=None

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0

        assert (self.d_inner / self.d_head) % 8 == 0, "requierement of causal_conv1d"

class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

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
    def __init__(self, config: Mamba2Config):
        super().__init__()

        self.mixer = Mamba2Block(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class Mamba2Block(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self.config = config        

        # [z, x, B, C, dt]
        d_in_proj = 2 * self.config.d_inner + 2 * self.config.n_groups * self.config.d_state + self.config.n_heads
        self.in_proj = nn.Linear(self.config.d_model, d_in_proj, bias=self.config.bias, **factory_kwargs)

        conv_dim = self.config.d_inner + 2 * self.config.n_groups * self.config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.config.conv_bias,
            kernel_size=self.config.d_conv,
            groups=conv_dim,
            padding=self.config.d_conv - 1,
            **factory_kwargs,
        )

        if self.config.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.config.conv_init, self.config.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.config.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.config.n_heads, self.config.d_head, self.config.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.config.n_heads, **factory_kwargs) * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt = torch.clamp(dt, min=self.config.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert self.config.A_init_range[0] > 0 and self.config.A_init_range[1] >= self.config.A_init_range[0]
        A = torch.empty(self.config.n_heads, dtype=torch.float32, device=self.config.device).uniform_(*self.config.A_init_range)
        A_log = torch.log(A).to(dtype=self.config.dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.config.n_heads, device=self.config.device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.config.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.config.d_inner, self.config.d_model, bias=self.config.bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """

        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.config.learnable_init_states else None
        dt_limit_kwargs = {} if self.config.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.config.dt_limit)

        if self.config.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.config.chunk_size,
                seq_idx=seq_idx,
                activation=self.config.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.config.d_head,
                ngroups=self.config.n_groups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.config.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.config.activation not in ["silu", "swish"]:
                xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)) # (B, L, self.d_inner + 2 * n_groups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.config.activation,
                ).transpose(1, 2)

            # split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.config.d_head),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.config.n_groups),
                rearrange(C, "b l (g n) -> b l g n", g=self.config.n_groups),
                chunk_size=self.config.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight
