"""

Universal language model, which accepts as its core a Mamba.
It has an embedding layer, and a LM head which maps the model output to logits.

"""

from typing import Union
import inspect
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import Mamba, MambaConfig, RMSNorm
from mambapy.mamba2 import Mamba2, Mamba2Config

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

# todo : comments, and source

class LM(nn.Module):
    def __init__(self, model_config: Union[MambaConfig, Mamba2Config], vocab_size: int, pad_vocab_size_multiple: int = None):
        super().__init__()

        if pad_vocab_size_multiple != None and (vocab_size % pad_vocab_size_multiple != 0):
            vocab_size += (pad_vocab_size_multiple - vocab_size % pad_vocab_size_multiple)

        self.config = model_config

        self.embedding = nn.Embedding(vocab_size, self.config.d_model, padding_idx=0)
        
        if isinstance(self.config, MambaConfig):
            self.mamba = Mamba(self.config)
        elif isinstance(self.config, Mamba2Config):
            self.mamba = Mamba2(self.config)
        else:
            raise NotImplementedError

        self.norm_f = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)

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
        
        elif self.config.mup and isinstance(self.config, Mamba2Config):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult))
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D', 'mixer.dt_bias']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param has not been filtered out for init. please check."

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
        x = self.mamba(x)
        x = self.norm_f(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult

        logits = self.lm_head(x)

        return logits
    
    def step(self, token, caches):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, caches = self.mamba.step(x, caches)
        x = self.norm_f(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult
        
        logits = self.lm_head(x)

        return logits, caches
    
    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=input_ids.device)) for _ in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, i], caches) # (batch_size, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (batch_size)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
                
        outputs = [tokenizer.decode(output.tolist()) for output in input_ids]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs
    
    # non-muP init (taken from llama2.c)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)

    # adaped from llama2.c
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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

        elif self.config.mup and isinstance(self.config, Mamba2Config):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D and A

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

# adapted from https://github.com/johnma2006/mamba-minimal
def from_pretrained(name: str):
    """
    Returns a model loaded with pretrained weights pulled from HuggingFace.

    Note :
    This only work with the state-spaces/mamba-XXX model family, because there is a pytorch_model.bin file in the HF repo.
    This is not the case of typical model saved on HF (like the state-spaces/mamba-XXX-hf model family).
    To load the state dict of such models, I think the only way is to load the model into a AutoModelForCausalLM, and then
    pass the state_dict to a MambaLM. I see no other way around unfrortunately (this is how it's done in jamba.py)

    Args:
        name: As of now, supports
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: a Mamba model configured with the proper parameters and initialized with the proper weights
    """

    try:
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
    except ImportError:
        print("The from_pretrained function pulls weights from HuggingFace and thus needs transformers to be installed (pip install transformers)")
        return

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))
                
    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
    # copy config data
    config_data = load_config_hf(name)
    config = MambaConfig(d_model=config_data['d_model'], n_layers=config_data['n_layers'])
    model = LM(config, config_data['vocab_size'])

    # copy weights
    state_dict = load_state_dict_hf(name)

    new_state_dict = {}
    for key in state_dict:
        if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
            new_key = key.replace('backbone.', '')
        else:
            new_key = key.replace('backbone', 'mamba')

        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)

    return model
