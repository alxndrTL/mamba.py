"""
Adapted from the example in https://github.com/graphcore-research/unit-scaling.
They use it to benchmark u-muP, a newer version of muP.
"""

import os
from typing import *

import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from mambapy.mamba import MambaConfig
from mambapy.mamba2 import Mamba2Config
from mambapy.lm import LM

# Config & helpers
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model
vocab_size = 256
depth = 4
head_size = 16
mlp_expansion = 2

# Training
n_steps = int(5e3)
warmup_steps = int(1e3)
batch_size = 16
sequence_length = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compile = True

dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
data = torch.frombuffer(bytearray("".join(dataset["text"]), encoding="utf8"), dtype=torch.uint8)
def batches():
    for _ in range(n_steps):
        yield torch.stack([
            data[i:i + sequence_length].to(device=device, dtype=torch.long)
            for i in torch.randint(0, len(data) - sequence_length, size=(batch_size,))
        ])

def lr_schedule(step: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    a = (step - warmup_steps) * torch.pi / (n_steps - warmup_steps)
    return torch.tensor(a).cos().mul(.5).add(.5)

def run_experiment(type_: Literal["SP", "μP"], width: int, lr: float) -> List[Dict[str, Any]]:
    if type_ == "μP":
        #config = MambaConfig(d_model=width, n_layers=depth, mup=True, mup_base_width=64, use_cuda=True)
        config = Mamba2Config(d_model=width, n_layers=depth, d_head=head_size, mup=True, mup_base_width=64)
        model = LM(config, vocab_size).to(device)
    elif type_ == "SP":
        #config = MambaConfig(d_model=width, n_layers=depth, mup=False, use_cuda=True)
        config = Mamba2Config(d_model=width, n_layers=depth, d_head=head_size, mup=False)
        model = LM(config, vocab_size).to(device)
    opt = model.configure_optimizers(weight_decay=0., learning_rate=torch.tensor(lr, dtype=torch.float, device=device), betas=(0.9, 0.95), device_type=device)

    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    def run_step(batch):
        opt.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits[..., :-1, :].flatten(end_dim=-2), batch[..., 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        opt.step()
        schedule.step()
        return loss

    log = []
    log2lr = torch.tensor(lr).log2().item()
    progress = tqdm(enumerate(batches()), desc=f"{type_:>4}, width={width}, lr=2^{log2lr:<5.0f}")
    for step, batch in progress:
        loss = run_step(batch)

        if loss.item()>6:
            print("eeee")
        log.append(dict(step=step, loss=loss.item()))
        if (step + 1) % 100 == 0:
            progress.set_postfix_str(f"loss = {loss.item():.2f}")
    return pd.DataFrame.from_dict(log).assign(type=type_, width=width, lr=lr)


# ---------------

filename = "mamba2.results.json"
fig_name = "mamba2.png"

if os.path.exists(filename):
    with open(filename, "r") as f:
        existing_results = pd.read_json(f)
else:
    existing_results = pd.DataFrame()


type_to_lr_range = {
    "SP": [2**n for n in range(-16, -10 + 1)],
    "μP": [2**n for n in range(-13, -4 + 1)],
}
new_results = pd.concat([
        run_experiment(type_=type_, width=width, lr=lr)
        for type_, lrs in type_to_lr_range.items()
        for width in [2048]
        for lr in lrs
]).reset_index(drop=True)
combined_results = pd.concat([existing_results, new_results]).reset_index(drop=True)
combined_results.to_json(filename)

combined_results["loss"] = combined_results["loss"].fillna(2.)
df_final = combined_results.groupby(["type", "width", "lr"])["loss"].apply(lambda g: min(g.iloc[-50:].mean(), 2.)).reset_index()

g = sns.relplot(data=df_final.pipe(lambda d: d.assign(width=d.width.apply(str))),
                y="loss", x="lr", hue="width", col="type",
                kind="line", facet_kws=dict(sharex=False), height=4)
for type_, ax in g.axes_dict.items():
    ax.set_title(type_)
    ax.set_xscale("log", base=2)

g.savefig(fig_name, dpi=600)
