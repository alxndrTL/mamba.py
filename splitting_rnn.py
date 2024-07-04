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

model = MyRNN(input_size=10, hidden_size=20, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(2, 100, 10) # (B, L, D)

chunk_size = 30
num_chunks = x.shape[1] // chunk_size
remainder = x.shape[1] % chunk_size

# classic forward
hidden = None
optimizer.zero_grad()

output, hidden = model(x, hidden)
loss = output.sum()
loss.backward()

print(model.fc.weight.grad.mean())
print(model.rnn.weight_ih_l0.grad.mean())
print(model.rnn.weight_hh_l0.grad.mean())
print(model.rnn.weight_ih_l1.grad.mean())
print(model.rnn.weight_hh_l1.grad.mean())

# segmented forward
hidden = None
optimizer.zero_grad()

print(f"seq len: {x.shape[1]}")
print(f"number of chunks: {num_chunks} and remainder: {remainder}")

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    chunk = x[:, start_idx:end_idx]
    
    output, hidden = model(chunk, hidden)
    loss = output.sum()
    loss.backward(retain_graph=True)

if remainder > 0:
    remainder_start_idx = num_chunks * chunk_size
    chunk = x[:, remainder_start_idx:]

    output, hidden = model(chunk, hidden)
    loss = output.sum()
    loss.backward()

print(model.fc.weight.grad.mean())
print(model.rnn.weight_ih_l0.grad.mean())
print(model.rnn.weight_hh_l0.grad.mean())
print(model.rnn.weight_ih_l1.grad.mean())
print(model.rnn.weight_hh_l1.grad.mean())
