"""
Universal language model, which accepts as its core a Transformer or a Mamba.

The Transformer is implemented in PyTorch and supports FlashAttention-2/
For Mamba, you have the choice : use mamba.py's pure PyTorch implementation (cf mamba/mamba.py) or use the CUDA implementation.
"""

from typing import Union
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import Mamba, MambaConfig, RMSNorm

# todo : inference function
# todo : comments, and source

# todo : join the two configure_optimizer func ?

class LM(nn.Module):
    def __init__(self, model_config: Union[MambaConfig, None], vocab_size: int):
        super().__init__()

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        
        if isinstance(self.config, MambaConfig):
            self.core = Mamba(self.config)
        else:
            raise NotImplementedError

        self.out_norm = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)

        self.lm_head = nn.Linear(self.config.d_model, vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

        if self.config.mup and isinstance(self.config, MambaConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_delta_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight', 'mixer.x_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    if 'mixer.dt_proj.weight' in pn:
                        std = self.config.dt_rank**-0.5 * self.config.dt_scale

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))
                elif 'mixer.x_BC_proj.weight' in pn:
                    torch.nn.init.zeros_(p[self.config.dt_rank:])
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if "bias" in pn:
                        torch.nn.init.zeros_(p)

        else:
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std/math.sqrt(2 * self.config.n_layers))

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        x = self.core(x)
        x = self.out_norm(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult

        logits = self.lm_head(x)

        return logits
        
    def forward_up_to(self, tokens, layer):
        # tokens : (B, L)
        # layer (1->n_layers): will stop the forward pass just after this layer

        # x : (B, L, D) activations after {layer}

        x = self.embedding(tokens)
        x = self.core(x, stop_at_layer=layer)

        return x
    
    # non-muP init (taken from llama2.c)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)

    # taken from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # any parameters that is 2D will be weight decayed, otherwise no. (i.e. all weight tensors in matmuls + embeddings decay, all biases and rmnsnorms don't)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def configure_optimizers_mup(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        if self.config.mup and isinstance(self.config, MambaConfig):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_delta_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight', 'mixer.x_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)

        return optimizer
