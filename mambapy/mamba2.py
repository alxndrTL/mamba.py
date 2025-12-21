"""

adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
It implements a Mamba2 model BUT still relies on the official Triton code for Mamba2.
(Coming soon hopefully is a full torch version, like in mamba.py and pscan.py)

Also, the file implements a caching mecanism, and a config similar to what's being done in mamba.py, as well as supports muP.

When passing an input of length 1 in the forward function, the model automatically routes the call to the step function.
This step function does one "classic" RNN-like step of computation, using the input and the cache provided. It returns the output as well as the new cache.
This is used for inference, to generate tokens one by one.

Also, this model supports prefilling a prompt and decoding token by token : this is just a mix of forwarding (in parallel) the input, and then decoding step by step.
In order to do that, we need the forward/parallel call (also used in training) to output the cache, then used to start the step by step decoding part.
To use this mode, you need to call Mamba2.forward() with your input (of shape (B, L, D)) as well as with a non-None cache (like full zeros, it doesnt matter, just not "None").
The forward call will thus return the output and the cache. From there, you can start your step by step decoding (see just above).

The cache is composed of two objects :
-h_cache: the last hidden state. Just like an RNN : you have to keep track of only the last h.
-conv_state: because Mamba2 uses a convolution over the time sequence, with a filter of length d_conv=4, you have to keep the last d_conv-1=3 inputs of that convolution to be able to run it provided a new input.

h_cache is of shape (B, n_heads, d_head, N) and is initialized at 0 (ie no starting hidden state, which is the default behavior in Mamba).
conv_state is of shape (B, EDN * 2*n_groups*, d_conv) and is initialized at 0

(B=batch_size, L=seq len, E = expand_factor, D=d_model, N=d_state)



(TODO: make it full pytorch (ie translate mamba_chunk_scan_combined in pytorch))

"""

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm

    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    
except ImportError:
    RMSNormGated, LayerNorm = None, None

    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None
    selective_state_update = None

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

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    chunk_size: int = 256
    use_mem_eff_path: bool = True
    dtype=None
    device=None

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0

        assert (self.d_inner / self.d_head) % 8 == 0, "requierement of causal_conv1d"

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, caches=None):
        # x : (B, L, D)

        # y : (B, L, D)

        if caches is None:
            caches = [None] * self.config.n_layers

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])

        if caches[0] == None:
            return x
        else:
            return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        
        self.config = config

        self.mixer = Mamba2Block(self.config)
        self.norm = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)

    def forward(self, x, cache=None):
        # x : (B, L, D)

        # output : (B, L, D)

        output, cache = self.mixer(self.norm(x), cache)
        output = output + x
        return output, cache
    
    def get_empty_cache(self, batch_size):
        h_cache = torch.zeros(batch_size, self.config.n_heads, self.config.d_head, self.config.d_state, device=self.mixer.in_proj.weight.device, dtype=self.mixer.in_proj.weight.dtype)
        conv_cache = torch.zeros(batch_size, self.mixer.conv1d.weight.shape[0], self.config.d_conv, device=self.mixer.conv1d.weight.device, dtype=self.mixer.conv1d.weight.dtype)
        return (h_cache, conv_cache)

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

        # todo : mup init + lr
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

    def forward(self, u, cache=None, seq_idx=None):
        """
        u: (B, L, D)
        Returns: out : same shape as u
        """

        batch, length, _ = u.shape

        return_cache = False
        if cache is not None and length > 1:
            cache = None
            return_cache = True
        
        if cache is not None:
            out, cache = self.step(u, cache)
            return out, cache

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
                return_final_states=return_cache,
                **dt_limit_kwargs,
            )

            if return_cache:
                # get h_cache from output
                out, h_cache = out

                # compute conv_cache with last d_conv entries of xBC
                _, xBC, _ = torch.split(zxbcdt, [self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)
                conv_cache = xBC[:, -self.config.d_conv:].transpose(1, 2) # (error if seqlen<d_conv)

                cache = (h_cache, conv_cache)

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
        return out, cache
    
    def step(self, u, cache):
        """
        u: (B, 1, D)
        cache: (h_cache, conv_cache)
        """
        
        h_cache, conv_cache = cache

        zxbcdt = self.in_proj(u.squeeze(1))  # (B, 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.config.d_inner - 2 * self.config.n_groups * self.config.d_state - self.config.n_heads) // 2
        z0, x0, z, xBC, dt = torch.split(zxbcdt, [d_mlp, d_mlp, self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)

        # conv step
        if causal_conv1d_update is None:
            conv_cache.copy_(torch.roll(conv_cache, shifts=-1, dims=-1)) # update state (B, D, W)
            conv_cache[:, :, -1] = xBC
            xBC = torch.sum(conv_cache * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1) # (B, D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=x.dtype)
        else:
            xBC = causal_conv1d_update(xBC, conv_cache, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, self.config.activation)
        x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
        A = -torch.exp(self.A_log.float()) # (n_heads)

        # SSM step
        if selective_state_update is None:
            assert self.config.n_groups == 1, "Only support ngroups=1 for this inference code path"
            # discretize A
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (B, n_heads)
            dA = torch.exp(dt * A)  # (B, n_heads)
            # discretize B
            x = rearrange(x, "b (h p) -> b h p", p=self.config.d_head)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            # compute one step
            h_cache.copy_(h_cache * rearrange(dA, "b h -> b h 1 1") + dBx)
            # compute output
            y = torch.einsum("bhpn,bn->bhp", h_cache.to(x.dtype), C)
            y = y + rearrange(self.D.to(x.dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
        
        else:
            A = repeat(A, "h -> h p n", p=self.config.d_head, n=self.config.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.config.d_head)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.config.d_head)
            D = repeat(self.D, "h -> h p", p=self.config.d_head)
            B = rearrange(B, "b (g n) -> b g n", g=self.config.n_groups)
            C = rearrange(C, "b (g n) -> b g n", g=self.config.n_groups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.config.d_head)
            
            y = selective_state_update(h_cache, x_reshaped, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)
            y = rearrange(y, "b h p -> b (h p)")
        
        #if self.rmsnorm:
        y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), (h_cache, conv_cache)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
