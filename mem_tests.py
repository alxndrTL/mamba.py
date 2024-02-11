import torch
import torch.nn as nn
import torch.utils.checkpoint

class MyModel(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.fc1 = nn.Linear(N, N, bias=False)
        self.fc2 = nn.Linear(N, N, bias=False)
        self.fc3 = nn.Linear(N, N, bias=False)
        self.fc4 = nn.Linear(N, N, bias=False)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, X):
        # X : (B, N)

        # Y : (B, N)

        """
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        """

        X = torch.utils.checkpoint.checkpoint(self.custom(self.fc1), X, use_reentrant=False)
        #X = torch.utils.checkpoint.checkpoint(self.custom(self.fc2), X, use_reentrant=False)
        #X = torch.utils.checkpoint.checkpoint(self.custom(self.fc3), X, use_reentrant=False)
        X = self.fc2(X)
        X = self.fc3(X)
        X = torch.utils.checkpoint.checkpoint(self.custom(self.fc4), X, use_reentrant=False)

        return X
    
device = "cuda"
B, N = 102400, 256

model = MyModel(N).to(device) # = 1MB

X = torch.randn(B, N, requires_grad=True, device="cuda") # = 100MB

torch.cuda.reset_peak_memory_stats(device)
initial_memory = torch.cuda.max_memory_allocated(device)

Y = model(X)

J = Y.sum()

peak_memory_fwd = torch.cuda.max_memory_allocated(device=device)

J.backward()

peak_memory_all = torch.cuda.max_memory_allocated(device=device) 

print(f"{initial_memory/(1024**2)}MB")
print(f"{peak_memory_fwd/(1024**2)}MB")
print(f"{(peak_memory_all)/(1024**2)}MB")
