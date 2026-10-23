import torch

from mambapy.lm import LM
from mambapy.mamba import MambaConfig


def test_generate_with_top_k_uses_configured_vocabulary_size():
    model = LM(MambaConfig(d_model=8, n_layers=1, d_state=2), vocab_size=16)
    prompt = torch.tensor([[1, 2]])

    generated = model.generate(prompt, num_tokens=1, sample=True, top_k=1)

    assert generated.shape == (1, 1)
