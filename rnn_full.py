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

B, L, D = 512, 1000, 512

model = MyRNN(input_size=D, hidden_size=D, output_size=D).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.randn(B, L, D).to("cuda")

# classic forward
hidden = None
optimizer.zero_grad()

output, hidden = model(x, hidden)
loss = output.sum()
loss.backward()

print("Done")
