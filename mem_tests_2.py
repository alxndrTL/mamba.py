import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

class MyModel(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.fc1 = nn.Linear(N, N, bias=False)
        self.fc2 = nn.Linear(N, N, bias=False)
        self.fc3 = nn.Linear(N, N, bias=False)
        self.fc4 = nn.Linear(N, N, bias=False)

    def forward(self, X):
        # X : (B, N)

        # Y : (B, N)

        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)

        """
        X = torch.utils.checkpoint.checkpoint(self.fc1, X, use_reentrant=False)
        X = torch.utils.checkpoint.checkpoint(self.fc2, X, use_reentrant=False)
        X = torch.utils.checkpoint.checkpoint(self.fc3, X, use_reentrant=False)
        X = torch.utils.checkpoint.checkpoint(self.fc4, X, use_reentrant=False)
        """
        return X
    
device = "cuda"
B, N = 102400, 256

model = MyModel(N).to(device) # = 1MB

X = torch.randn(B, N, requires_grad=True, device="cuda") # = 100MB

torch.cuda.reset_peak_memory_stats(device)
initial_memory = torch.cuda.max_memory_allocated(device)

# number of checkpoint segments
segments = 2

# the modules should be in the order the model should be executed
modules = [module for _, module in model._modules.items()]

out = checkpoint_sequential(modules, segments, X)

J = out.sum()

peak_memory_fwd = torch.cuda.max_memory_allocated(device=device)

J.backward()

peak_memory_all = torch.cuda.max_memory_allocated(device=device) 

print(f"{initial_memory/(1024**2)}MB")
print(f"{peak_memory_fwd/(1024**2)}MB")
print(f"{(peak_memory_all)/(1024**2)}MB")
