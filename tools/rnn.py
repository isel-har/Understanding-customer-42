import torch.nn as nn
import numpy as np
import torch
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout_rate=0.3
        # device,
    ):
        super(RNN, self).__init__()
        
        # self.device=device
        self.rnn = nn.RNN(
            input_size=input_size,   # 300 (embedding dim)
            hidden_size=hidden_size, # e.g. 128
            num_layers=num_layers,   # e.g. 2
            batch_first=True,        # input: (batch, seq, feature)
            nonlinearity='tanh',     # 'tanh' or 'relu'
            dropout=dropout_rate          # dropout between layers
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)#.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)#.to(self.device)
        out, hidden = self.rnn(x, h0)
        out = out[:, -1, :]          # → (263, 128)
        out = self.fc(out)           # → (263, num_classes)
        return out
