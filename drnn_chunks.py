import torch
import torch.nn as nn
import torch.optim as optim

from rnn import RNNConfig, RNN

# 1000 = no OOM
B, L, D = 512, 1024, 64

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

config = RNNConfig(d_model=D, n_layers=2)
model = RNN(config).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(B, L, D).to("cuda")
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

chunk_size = 100
num_chunks = x.shape[1] // chunk_size
remainder = x.shape[1] % chunk_size

# segmented forward
hidden = [torch.zeros(B, D, device="cuda"), torch.zeros(B, D, device="cuda")]
optimizer.zero_grad()

print(f"seq len: {x.shape[1]}. number of chunks: {num_chunks} and remainder: {remainder}")

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunk = x[:, start_idx:end_idx]
    
    output = model(chunk, hidden)
    output_b = model(chunk, hidden, stop_at_layer=1)
    hidden = [output_b[:, -1], output[:, -1]]

    loss = output.sum()
    loss.backward(retain_graph=not(i==num_chunks-1 and remainder==0))
    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    chunk = x[:, remainder_start_idx:]

    output = model(chunk, hidden)
    hidden = output[:, -1]

    loss = output.sum()
    loss.backward()

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

print("Done")

print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
