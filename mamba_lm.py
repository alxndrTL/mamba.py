from dataclasses import dataclass, fields, asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import Mamba, MambaConfig, RMSNorm


"""

Encapsulates a Mamba model as language model. It has an embedding layer, and a LM head which maps the model output to logits.

"""

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
    
def from_pretrained(name: str):
    # ex : 'state-spaces/mamba-130m'

    # adapted from https://github.com/johnma2006/mamba-minimal

    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

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

    return model

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

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits

    # adapted from https://github.com/johnma2006/mamba-minimal
    def generate(self, tokenizer, prompt: str, n_tokens_to_gen: int = 50, sample: bool = True, top_k: int = 40):
        self.eval()
    
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device)
        
        for _ in range(n_tokens_to_gen):
            with torch.no_grad():
                indices_to_input = input_ids
                next_token_logits = self(indices_to_input)[:, -1]
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            if top_k is not None:
                values, _ = torch.topk(probs, k=top_k)
                probs[probs < values[:, -1, None]] = 0
                probs = probs / probs.sum(axis=1, keepdims=True)
            
            if sample:
                next_indices = torch.multinomial(probs, num_samples=1)
            else:
                next_indices = torch.argmax(probs, dim=-1)[:, None]
            
            input_ids = torch.cat([input_ids, next_indices], dim=1)

        output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

        self.train()
        
        return output_completions
