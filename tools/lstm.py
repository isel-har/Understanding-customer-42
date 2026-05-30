import torch
import torch.nn as nn
import numpy as np
import random


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False   # set True for BiLSTM
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]   # last time step → (263, 128)
        out = self.fc(out)    # → (263, num_classes)
        return out