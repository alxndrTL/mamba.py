from dataclasses import dataclass, fields, asdict
import json

import mlx.core as mx
import mlx.nn as nn

from mamba_mlx import Mamba, MambaConfig
from misc import topk
from utils import load_config_hf, load_state_dict_hf

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

class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = nn.RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight #TODO this does not really tie the weights, investigate

    def __call__(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

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

        logits = self.lm_head(x)

        return logits, caches
    
    def generate(self, tokenizer, prompt: str, n_tokens_to_gen: int = 50, sample: bool = True, temperature: float = 1.0, top_k: int = None):
        self.eval()

        input_ids = mx.array(tokenizer(prompt, return_tensors='np').input_ids) # (1, tokens_prompt) # (1, num_tokens)

        # caches is a list of cache, one per layer
        # cache is composed of : the hidden state, and the last d_conv-1 inputs
        # the hidden state because the update is like an RNN
        # the last d_conv-1 inputs because they are used in a 1d convolution (usually d_conv=4 so this is not large)
        caches = [(None, mx.zeros([1, self.config.d_conv-1, self.config.d_inner])) for _ in range(self.config.n_layers)]

        for i in range(input_ids.shape[1] + n_tokens_to_gen - 1):
            next_token_logits, caches = self.step(input_ids[:, i], caches) # (1, vocab_size), caches

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.shape[1]:
                
                if top_k is not None:
                    values = topk(next_token_logits, k=top_k) # (1, k) ordered from lowest to biggest
                    mask = next_token_logits < (values[:, 0, None])
                    next_token_logits = mx.where(mask, -5000, next_token_logits) # TODO -mx.inf is problematic for now

                if sample and temperature > 0:
                    next_token = mx.random.categorical(next_token_logits * (1/temperature), num_samples=1)
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1)[:, None]

                input_ids = mx.concatenate([input_ids, next_token], axis=1)

        output = [tokenizer.decode(output.tolist()) for output in input_ids][0]

        self.train()

        return output
    
    @staticmethod
    def from_pretrained(name: str):
        """
        Returns a model loaded with pretrained weights pulled from HuggingFace.

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

        import os
        import numpy as np
        from mlx.utils import tree_unflatten

        from utils import map_mambassm_torch_to_mlx

        # copy config data
        config_data = load_config_hf(name)
        config = MambaLMConfig(d_model=config_data['d_model'], n_layers=config_data['n_layer'], vocab_size=config_data['vocab_size'])

        model = MambaLM(config)

        # copy weights
        filename = name.split('/')[-1] + '.mlx.npz'

        if not os.path.exists(filename):
            state_dict = load_state_dict_hf(name)
            mlx_state_dict = map_mambassm_torch_to_mlx(state_dict)

            np.savez(filename, **mlx_state_dict)

        model.update(tree_unflatten(list(mx.load(filename).items())))

        return model
