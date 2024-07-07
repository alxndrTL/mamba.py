import torch

from rnn import RNNConfig, RNN

torch.cuda.reset_peak_memory_stats(device=None)
torch.manual_seed(123456)

B, L, D = 32, 1024, 64

config = RNNConfig(d_model=D, n_layers=2, dropout=0.)
model = RNN(config).to("cuda")

x = torch.randn(B, L, D, device="cuda")
x.requires_grad = True

hs = model(x, [torch.zeros(B, D, device="cuda"), torch.zeros(B, D, device="cuda")])
J = hs.sum()
J.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")

print("Done")

print(model.layers[0].cell.fc.weight.grad.mean())
print(model.layers[0].cell.fc.bias.grad.mean())
print(model.layers[1].cell.fc.weight.grad.mean())
print(model.layers[1].cell.fc.bias.grad.mean())
