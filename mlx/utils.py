import json
from typing import Union

import numpy as np
import mlx.core as mx
import torch

from misc import torch_to_mlx_depthwise_weights

# TODO : map_mlx_to_mambapy_torch

def load_config_hf(model_name):
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))
                
def load_state_dict_hf(model_name):
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

def map_mambapy_torch_to_mlx(torch_state_dict):
    new_state_dict = {}
    for key, value in torch_state_dict.items():

        # from torch to mlx, we need to convert the conv weights (see misc.py for explanations)
        if 'conv1d.weight' in key:
            value = torch_to_mlx_depthwise_weights(value)

        if 'conv1d' in key:
            key = key.replace('conv1d', 'conv1d.conv1d')

        if value.type() == 'torch.BFloat16Tensor':
            new_state_dict[key] = value.half().numpy()
        else:
            new_state_dict[key] = value.numpy()

    return new_state_dict

def map_mambassm_torch_to_mlx(torch_state_dict):
    # convert mambassm to mambapy
    new_state_dict = {}
    for key in torch_state_dict:
        if key == 'backbone.embedding.weight' or key == 'backbone.norm_f.weight':
            new_key = key.replace('backbone.', '')
        else:
            new_key = key.replace('backbone', 'mamba')

        new_state_dict[new_key] = torch_state_dict[key]

    # convert mambapy to mlx
    return map_mambapy_torch_to_mlx(new_state_dict)

"""
# todo : doesnt work, because MambaConfig and MambaLMConfig are not the ones defined in mamba.py and mamba_lm.py
def mambapy_torch_to_mlx(torch_state_dict, config: Union[MambaConfig, MambaLMConfig]):
    mlx_state_dict = map_mambapy_torch_to_mlx(torch_state_dict)

    if isinstance(config, MambaConfig):
        model = Mamba(config)
    else:
        model = MambaLM(config)

    np.savez("weights.mlx.npz", **mlx_state_dict) # TODO name with config?
    model.update(tree_unflatten(list(mx.load("weights.mlx.npz").items())))

    # todo : name the file according to config
    # todo : check if file already exists

    return model
"""
