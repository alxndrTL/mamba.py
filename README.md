# mamba.py 🐍 : a simple parallel scan implementation
A straightfoward implementation of [Mamba](https://arxiv.org/abs/2312.00752) in PyTorch with a simple parallel scan implementation, offering an major speedup over a a sequential implementation.
It combines the ease of read with good performances.

![speed comparison](assets/speed_comparison.png)

This graph shows the training time (forward and backward pass) of a single Mamba layer (`d_model=16, d_state=16`) using 3 different methods : `CUDA`, which is the official [Mamba implementation](https://github.com/state-spaces/mamba), `mamba.py`, which is this repo, and `sequential`, which is a  sequential implementation of the selective scan.

This repo contains a simple and readable code implementing the [Mamba](https://arxiv.org/abs/2312.00752) architecture in pure PyTorch. Its primary goal is educational.

<p align="center">
    <img src="assets/logo.png" alt="a python and a mamba" width="300" height="300" alt="python mamba"/>
</p>

<u>The repo is organized as follows : </u>
- `pscan.py` : a PyTorch implementation of Blelloch's parallel scan
- `mamba.py` : the Mamba model, as described in the [paper](https://arxiv.org/abs/2312.00752). It is numerically equivalent (initialization, forward and backward pass).
- `mamba_lm.py` : encapsulates a Mamba model in order to use it as a language model
- `📁 docs` : a folder containing annotated explanations about the code, focusing on the parallel scan
- `📁 examples` : two examples of how to use the Mamba model.

## Usage

The most basic usage is to use the `Mamba` object ([mamba.py](mamba.py)), which implements a simple Mamba model given a configuration.
No embedding, no head : input is `(B, L, D)` and output is `(B, L, D)` as well.

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

The class `MambaLM` ([mamba_lm.py](mamba_lm.py)) builds on the `Mamba` object and offers a classic API for language models. It can be used as follows :

```
from mamba_lm import MambaLM, MambaLMConfig

config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=32000)
model = MambaLM(config)

x = torch.randint(high=32000, size=(16, 64))
logits = model(x) # (B, L, vocab_size)
```

It simply encapsulates a `Mamba` object with an embedding layer, a final normalization and a language modeling head.

## Examples
There are two basics examples available :
- `example_llm.ipynb` : load a Mamba model with pretrained weights (from 130M to 2.8B from HuggingFace)
- `example_e2e_training.ipynb` : an end-to-end training example where a Mamba model is employed as a world model for a simple 3-3 grid game (training is not completed, the model should be larger).

## Performances
This section provides a more comprehensive performance comparison between `mamba.py` and the official Mamba implementation.
Overall, as the first graph of this file shows, both have approximately the same asymptotic performance with respect to the sequence length. You can think as `mamba.py` as a regular Transformer implementation, while the official Mamba implementation is more like FlashAttention v1. Both have their owns advantages.

That being said, does the two implementations have the same asymptotic performances with respect to the other parameters ?

##### `d_model` asymptotic performances
<p align="center">
    <img src="assets/training_vs_d_model.png" alt="a python and a mamba" 
    width="800" height="413" alt="python mamba"/>
</p>

We can see that both implementations behave the same as we increase `d_model`. The gap between the two stays roughly the same. (`mamba.py` is overall ~2x slower)

##### `d_state` asymptotic performances
<p align="center">
    <img src="assets/training_vs_d_state.png" alt="a python and a mamba" 
    width="800" height="413" alt="python mamba"/>
</p>

This graph is important. We see that here, the asymptotic performance is not the same as we increase `d_state`. For a reminder, `d_state`, or $N$ in the paper, is the state expansion factor : each channel of the input is expanded into $N$ channels of the hidden state.

Does it matter in practice ? As of now, all the pretrained Mamba models (up to 2.8B parameters) used `d_state=16`, so this change of performance over `d_state` isn't important in this case. As `d_state` is not something that is supposed to grow (contrary to the seq length or `d_model`), this isn't a catastrophic result, but something to consider.

However, it is interesting to relate this observation with the claim made by Albert Gu and Tri Dao [Mamba paper](https://arxiv.org/abs/2312.00752) : <i>The main idea is to leverage properties of modern accelerators (GPUs) to <b>materialize the state ℎ only in more efficient levels of the memory hierarchy.</b></i>
They also describe (Annex D) the main data movements of their selective scan : working mainly in SRAM, they can reduce the memory reads/writes by a factor of $O(N)$. This explains what we're seeing here.

With `d_state=16` (as in `state-spaces/mamba-2.8b-slimpj`), the gap between the two is relatively small, but with `d_state=64` (currently not used in any models), the gap widens. (note the OOM on the second graph)

<p align="center">
    <img src="assets/training_vs_seqlen_d_state_var.png" alt="a python and a mamba" 
    width="1152" height="240" alt="python mamba"/>
</p>

All the previous graph were computed with a batch size of 1, on a A100 80GB.
It is a measure of both the forward and backward pass of a single Mamba block.

## Sources and where to learn more
- the [Mamba paper](https://arxiv.org/abs/2312.00752) : describes the Mamba architecture as implemented in this repo, which allows to model sequences in linear time.
- the [Mamba implementation](https://github.com/state-spaces/mamba), which is written in PyTorch but uses a parallel scan written in CUDA. This is the version that is the fastest. 
- [a minimal PyTorch implementation of Mamba](https://github.com/johnma2006/mamba-minimal), which implements the scan operation as a sequential loop (its performance are a bit worse than the 'sequential' line in the first graph). This code closely follows [this file](https://github.com/state-spaces/mamba/blob/da2626b5a5f347a8e844ac5e96a2cbcde3c34abb/mamba_ssm/modules/mamba_simple.py) from the officile Mamba implementation, but replaces the CUDA convolution with `torch.nn.Conv1d`, and the selective scan written in CUDA with a sequential loop. The code of this repo follows the structure of these 2 files.
- [Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf), by Guy E. Blelloch (1993).
- [Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057) : applies a parallel scan over the sequence in order to get rid of the sequential for-loop.
- x.com/fchollet : original pscan implementation.

## TODOs
- docs
- ~~more tests with an increased `d_model` (add a Performances section)~~
- ~~a step function, used for (auto-regressive) inference.~~
- a training function, similar to [llama2.c](https://github.com/karpathy/llama2.c)

perfs :
- unfold the for-loops in `pscan.py` to achieve better performance (see [François Fleuret's pscan](https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=mygptrnn.git;a=blob;f=pscan.py;h=0bb0d145bf9c6c82115956c8ce1e6a063e56e747;hb=HEAD)) (although this will sacrifice readability of bit)
- write a reverse parallel scan specifically for the backward pass. (For now, we have to flip the array before and after the scan).
- use torch.compile(). As far as I tested, it doesn’t work for now. It seems it isn’t happy with the custom PScan autograd function. Need to investigate. <b>(see [PR#1](https://github.com/alxndrTL/mamba.py/pull/1))</b>
