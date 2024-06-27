from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.onnx.mamba_onnx import Mamba, MambaConfig, RMSNorm

"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

# TODO generate function : batch size != 1 ? (for now B=1)
# TODO generate function : top-p sampling

@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

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
    config = MambaLMConfig(d_model=config_data['d_model'], n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])

    model = MambaLM(config)

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

    return model #, config

class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def init_caches(self):
        # hs will be initialized to zeros, so do inputs
        hs = torch.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_state, device=next(self.parameters()).device)
        # inputs size would be like this
        inputs = torch.zeros(self.config.n_layers, 1, self.config.d_inner, self.config.d_conv-1, device=next(self.parameters()).device)
        
        return hs, inputs
        
    def forward(self, token, hs, inputs):
        # token : (B)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # logits : (B, vocab_size)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        x = self.embedding(token)

        x, hs, inputs = self.mamba.step(x, hs, inputs)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits, hs, inputs
    