# mamba.py 🐍 : a simple parallel scan implementation
A straightfoward implementation of [Mamba](https://arxiv.org/abs/2312.00752) in PyTorch with a simple parallel scan implementation, offering an major speedup over a a sequential implementation.
It combines the ease of read with good performances.

![speed comparison](img/speed_comparison.png)

This repo contains a simple and readable code implementing the [Mamba](https://arxiv.org/abs/2312.00752) architecture in pure PyTorch. Its primary goal is educational.

<p align="center">
    <img src="img/logo.png" alt="Image Description" width="300" height="300" alt="python mamba"/>
</p>

<u>The repo is organized as follows : </u>
- ```pscan.py``` : a PyTorch implementation of Blelloch's parallel scan
- ```mamba.py``` : the Mamba model, as described in the [paper](https://arxiv.org/abs/2312.00752). It is numerically equivalent (initialization, forward and backward pass).
- ```mamba_lm.py``` : encapsulates a Mamba model in order to use it as a language model
- ```📁 docs``` : a folder containing annotated explanations about the code, focusing on the parallel scan
- ```📁 examples``` : two examples of how to use the Mamba model.

## Usage

The most basic usage is to use the ```Mamba``` object ([mamba.py](mamba.py)), which implements a simple Mamba model given a configuration.
No embedding, no head : input is ```(B, L, D)``` and output is ```(B, L, D)``` as well.

```
import torch
from mamba import Mamba, MambaConfig

config = MambaConfig(d_model=16, n_layers=2)
model = Mamba(config)

B, L, D = 2, 64, 16
x = torch.randn(B, L, D)
y = model(x)

assert y.shape == x.shape
```

The class ```MambaLM``` ([mamba_lm.py](mamba_lm.py)) builds on the ```Mamba``` object and offers a classic API for language models. It can be used as follows :

```
from mamba_lm import MambaLM, MambaLMConfig

config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=32000)
model = MambaLM(config)

x = torch.randint(high=32000, size=(16, 64))
logits = model(x) # (B, L, vocab_size)
```

It simply encapsulates a ```Mamba``` object with an embedding layer, a final normalization and a language modeling head.

## Examples
There are two basics examples available :
- ```example_llm.ipynb``` : load a Mamba model with pretrained weights (from 130M to 2.8B from HuggingFace)
- ```example_e2e_training.ipynb``` : an end-to-end training example where a Mamba model is employed as a world model for a simple 3-3 grid game (training is not completed, the model should be larger).


## Sources and where to learn more
- the [Mamba paper](https://arxiv.org/abs/2312.00752) : describes the Mamba architecture as implemented in this repo, which allows to model sequences in linear time.
- the [Mamba implementation](https://github.com/state-spaces/mamba), which is written in PyTorch but uses a parallel scan written in CUDA. This is the version that is the fastest. 
- [a minimal PyTorch implementation of Mamba](https://github.com/johnma2006/mamba-minimal), which implements the scan operation as a sequential loop. This code closely follows [this file](https://github.com/state-spaces/mamba/blob/da2626b5a5f347a8e844ac5e96a2cbcde3c34abb/mamba_ssm/modules/mamba_simple.py) from the officile Mamba implementation, but replaces the CUDA convolution with ```torch.nn.Conv1d```, and the selective scan written in CUDA with a sequential loop. The code of this repo closely follows these 2 files.
- [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf), by Guy E. Blelloch (1993).
- [Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057) : applies a parallel scan over the sequence in order to get rid of the sequential for-loop.
- x.com/fchollet : original pscan implementation.

## TODOs
- docs
- a step function, used for (auto-regressive) inference.
- write a reverse parallel scan specifically for the backward pass. (For now, we have to flip the array before and after the scan).
- use torch.compile(). As far as I tested, it doesn’t work for now. It seems it isn’t happy with the custom PScan autograd function. Need to investigate.
