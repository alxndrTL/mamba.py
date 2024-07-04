import torch
import torch.nn as nn
import torch.optim as optim

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden) # (B, L, hidden_size), (num_layers, B, hidden_size)
        out = self.fc(out)
        return out, hidden  

# 1000 = no OOM
B, L, D = 512, 250, 512

torch.cuda.reset_peak_memory_stats(device=None)
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

model = MyRNN(input_size=D, hidden_size=D, output_size=D).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")
x = torch.randn(B, L, D).to("cuda")
print(f"gpu used {torch.cuda.memory_allocated(device=None) / (1024**2)} MB")

chunk_size = 100
num_chunks = x.shape[1] // chunk_size
remainder = x.shape[1] % chunk_size

# segmented forward
hidden = None
optimizer.zero_grad()

print(f"seq len: {x.shape[1]}. number of chunks: {num_chunks} and remainder: {remainder}")

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunk = x[:, start_idx:end_idx]
    
    output, hidden = model(chunk, hidden)
    loss = output.sum()
    loss.backward(retain_graph=not(i==num_chunks-1 and remainder==0))

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    chunk = x[:, remainder_start_idx:]

    output, hidden = model(chunk, hidden)
    loss = output.sum()
    loss.backward()

print("Done")

print(f"gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
