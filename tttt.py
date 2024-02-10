import torch
import torch.nn as nn

import pandas as pd

from mem_utils import log_mem, plot_mem

#from pscan import pscan
from pscan_mem import pscan

class PScanModel(nn.Module):
    def __init__(self, ):
        super().__init__() 

    def forward(self, A, X):
        return pscan(A, X)

device = "cuda"

B, L, D, N = 1, 512, 32, 16

model = PScanModel()

A = torch.randn(B, L, D, N, requires_grad=True).to("cuda")
X = torch.randn(B, L, D, N, requires_grad=True).to("cuda")

mem_log = []

try:
    mem_log.extend(log_mem(model, input, exp='baseline'))
except Exception as e:
    print(f'log_mem failed because of {e}')

df = pd.DataFrame(mem_log)

plot_mem(df, exps=['baseline'], output_file=f'baseline_memory_plot.png')
