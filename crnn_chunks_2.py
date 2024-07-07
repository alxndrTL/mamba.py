import torch
from rnn import RNNConfig, RNN

# Reset GPU memory statistics and set the random seed
torch.cuda.reset_peak_memory_stats(device=None)
torch.manual_seed(123456)

# Define batch size, sequence length, and model dimension
B, L, D = 32, 1024, 64

# Configure and initialize the RNN model
config = RNNConfig(d_model=D, n_layers=2, dropout=0.)
model = RNN(config).to("cuda")

# Generate random input data
x = torch.randn(B, L, D, device="cuda")
x.requires_grad = True

# Define the chunk size (sequence length per chunk)
chunk_size = 128

# Initialize hidden states
h0_l1 = torch.zeros(B, D, device="cuda")
h0_l2 = torch.zeros(B, D, device="cuda")

# Process the input in chunks
#hss = []
for i in range(0, L, chunk_size):
    x_chunk = x[:, i:i+chunk_size, :]

    hs = model(x_chunk, [h0_l1, h0_l2])
    hs_l1 = model(x_chunk, [h0_l1, h0_l2], stop_at_layer=1)

    h0_l1 = hs_l1[:, -1]
    h0_l2 = hs[:, -1]

    output = hs.sum()
    output.backward(retain_graph=not(i==L-1))

    print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

    #hss.append(hs)

#hss = torch.cat(hss, dim=1)

#J = hss.sum()
#J.backward()

# Print GPU memory usage
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
print(f"max gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")

print("Done")

print(model.layers[0].cell.fc.weight.grad.mean())
print(model.layers[0].cell.fc.bias.grad.mean())
print(model.layers[1].cell.fc.weight.grad.mean())
print(model.layers[1].cell.fc.bias.grad.mean())