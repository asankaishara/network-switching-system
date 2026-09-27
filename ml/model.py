#	ml/model.py
"""Phase	3:	the	LSTM	forecaster."""
import torch
import torch.nn as nn

class LatencyLSTM(nn.Module):
    def __init__(self, n_features, hidden=48, layers=1, n_outputs=9):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, n_outputs))

    def forward(self, x):                 # x: (batch, timesteps, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # use the last timestep's hidden state