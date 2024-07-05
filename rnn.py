import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class RNNConfig:
    d_model: int # D or d_model in comments
    n_layers: int

    type: str = "rnn" # "rnn" ou "gru"  

    dropout: float = 0.1
    bias: bool = False
    norm_eps: float = 1e-5

class RNN(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()

        self.config = config

        self.in_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([RNNLayer(config) for _ in range(config.n_layers)])

    def forward(self, X, h0s, stop_at_layer: int = None):
        # X : (B, L, D)
        # h0s : [(B, d_model)] one h0 per layer
        # stop_at_layer (1 -> n_layers) : if set, will return the activations after the {layer}-th layer

        # Y : (B, L, D)

        _, T, _ = X.size()

        X = self.in_dropout(X)

        for i, layer in enumerate(self.layers):
            X = layer(X, h0s[i]) # (B, L, d_model)

            if stop_at_layer == i+1:
                return X
        
        return X

class RNNLayer(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()

        self.cell = RecurrentUnit(config) if config.type == "rnn" else GatedRecurrentUnit(config)
        
    def forward(self, X, h0):
        # X : (B, L, D)
        # h0 : (B, d_model)

        # Y : (B, L, D)
        
        B, L, _ = X.size()

        H_prev = h0 # (B, d_model)
        hiddens = []
        for t in range(L):
            X_t = X[:, t, :] # (B, d_model)
            H_t = self.cell(X_t, H_prev) # (B, d_model)
            H_prev = H_t
            hiddens.append(H_t)

        hidden = torch.stack(hiddens, 1) # (B, L, d_model)
        return hidden

class RecurrentUnit(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.fc = nn.Linear(2*config.d_model, config.d_model)

    def forward(self, x, h_prev):
        h = self.fc(torch.cat([x, h_prev], dim=1))
        h = F.tanh(h)
        return h
    
class GatedRecurrentUnit(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.fc_z = nn.Linear(2*config.d_model, config.d_model)
        self.fc_r = nn.Linear(2*config.d_model, config.d_model)
        self.fc_h_hat = nn.Linear(2*config.d_model, config.d_model)

    def forward(self, x, h_prev):
        z = F.sigmoid(self.fc_z(torch.cat([x, h_prev], dim=1)))
        r = F.sigmoid(self.fc_r(torch.cat([x, h_prev], dim=1)))
        h_hat = F.tanh(self.fc_h_hat(torch.cat([x, r * h_prev], dim=1))) # * = torch.mul
        h = (1 - z) * h_prev + z * h_hat

        # r permet de remove/reset des infos/channels de h_prev
        # on utilise alors ce hidden state comme candidat pour le suivant, h_hat
        # z permet de controler quels channels (de l'hidden state) update

        return h
