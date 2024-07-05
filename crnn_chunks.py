import torch

from rnn import RNNConfig, RNN

torch.cuda.reset_peak_memory_stats(device=None)
torch.manual_seed(123456)

B, L, D = 512, 1024, 64

config = RNNConfig(d_model=D, n_layers=2, dropout=0.)
model = RNN(config).to("cuda")

x = torch.randn(B, L, D, requires_grad=True, device="cuda")

chunk_size = 100 # best is power of 2 minus 1
num_chunks = L // chunk_size
remainder = L % chunk_size

print(f"number of chunks: {num_chunks} and remainder: {remainder}")

last_hidden = [torch.zeros(B, D, device="cuda"), torch.zeros(B, D, device="cuda")]

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    X_chunk = x[:, start_idx:end_idx]
    
    hs_chunk = model(X_chunk, last_hidden)
    hs_chunk_l1 = model(X_chunk, last_hidden, stop_at_layer=1)

    last_hidden = [hs_chunk_l1[:, -1], hs_chunk[:, -1]]

    loss = hs_chunk.sum()
    loss.backward(retain_graph=not(i==num_chunks-1 and remainder==0))
    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    X_chunk = x[:, remainder_start_idx:]

    hs_chunk = model(X_chunk, last_hidden)

    loss = hs_chunk.sum()
    loss.backward()
    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
print("Done")

print(model.layers[0].cell.fc.weight.grad.mean())
print(model.layers[0].cell.fc.bias.grad.mean())
print(model.layers[1].cell.fc.weight.grad.mean())
print(model.layers[1].cell.fc.bias.grad.mean())