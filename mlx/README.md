# MLX implementation of Mamba 🐍

This folder contains a full MLX implementation of [Mamba](https://arxiv.org/abs/2312.00752), which allows to train and do inference with Mamba models using an Apple silicon equiped Mac.
Both the <b>forward and backward pass</b> are numerically equivalent to the PyTorch code from `mamba.py`, as well as to the official [Mamba implementation](https://github.com/state-spaces/mamba).

<p align="center">
    <img src="assets/mamba_mlx.png" alt="a python and a mamba" width="400" height="400" alt="python mamba"/>
</p>

<u>The folder is organized as follows : </u>
- `pscan_mlx.py` : a MLX implementation of Blelloch's parallel scan.
- `mamba_mlx.py` : the Mamba model, as described in the [paper](https://arxiv.org/abs/2312.00752). It is numerically equivalent (initialization, forward and backward pass).
- `mamba_lm_mlx.py` : encapsulates a Mamba model in order to use it as a language model.
- `📁 scripts` : example scripts to play around with Mamba.

# Quickstart
First, you can clone the repo :

```
git clone https://github.com/alxndrTL/mamba.py.git
cd mamba.py/mlx
```

If you want to do inference with <b>pretrained models</b> (from 130M to 2.8B parameters), you can simply do :

```
cd scripts
python3 generate.py --prompt="Mamba is a type of" --hf_model_name="state-spaces/mamba-130m" --n_tokens=100
```

It will download the specified model from [HuggingFace](https://huggingface.co/state-spaces), convert it (and save it to disk) to run with MLX, and stream generated words.
As of now, you can choose from :

```
state-spaces/mamba-130m
state-spaces/mamba-370m
state-spaces/mamba-790m
state-spaces/mamba-1.4b
state-spaces/mamba-2.8b
state-spaces/mamba-2.8b-slimpj
```

As of today, only full precision inference is supported. On an M2 Pro (16GB), the 790M model runs at ~30tok/s.

Unlike the Transformers, inference doesn't depend on the sequence length, so we just have to carry along a hidden state 😎 (and the last `d_conv-1` inputs, where `d_conv` is usually 4).

As of now, `generate.py` is the only available script. But you can train the model using your own script, just like you would with a Transformer.

# About
Mamba is a new state-space model that is able to do sequence modeling - just like Transformers do.
While Transformers use attention to flow information through time, Mamba uses a simple hidden state, just like RNNs. It has the benefit of a constant inference time wrt. sequence length.
What is important to know is that while it uses a hidden state that is updated sequentially through time :

$$
h_t = A h_{t-1} + Bx_t
$$

all the $h_t$ can actually be computed <b>in parallel</b>, thanks to an algorithm named the <b>parallel scan</b>, implemented in `pscan_mlx.py` in MLX.
You can learn more about this algorithm and its implementation in `docs/pscan.ipynb` at the root of this repo. 
As you can see on graph shown on the landing page of this repo, the naive sequential implementation is way slower than implementations than use this parallel scan.

<b>However</b>, it's important to note that while the parallel scan gives correct computations with MLX, it's slow, so slow that it is sometimes actually harmful to use it.
<b>Why ?</b> It is not yet clear. When translating the algorithm from PyTorch to MLX, a little modification is needed : at each iteration, we need to write back to our original arrays the numbers we computed. This is because MLX doesn't have views implemented (yet?). (see [this issue](https://github.com/ml-explore/mlx/issues/466)). I thus switched to a version which only uses slicing (see `pscan_mlx.py` for more details), but the performances are still lacking behind the sequential version (should be orders of magnitude faster).

But, MLX is not even 2 months old :)
I will closely follow MLX development to watch for potential upgrades of this MLX implementation.

# Why [mamba.py](../) in MLX ?
While the primary goal of the PyTorch version is educational, this implementation (with a performing parallel scan) could power future fine-tuning scripts. We are early, as there is still not much resources about fine-tuned Mamba models (see [this](https://github.com/havenhq/mamba-chat)).

Also, the more people play around and train Mamba models, the more we will be able to know better its strengths and limits, allowing us to compare it against its "competitors" (Based, RWKV, StripedHyena, or the Transformer).

And finally, it was a great exercise for me, after having implemented Mamba in PyTorch and not knowing MLX.

# TODOs
- add more ready-to-go scripts (training and <b>fine-tuning</b>)
- support for mixed precision training ? (see [this](https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file#precision) from the official Mamba implementation)
- set device (cpu and gpu) (see [A Simple Example](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html#a-simple-example) from the MLX docs)
- see TODOs of the PyTorch versions
- watch out for new MLX updates ;)

Feel free to contribute !