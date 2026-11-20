import torch
import torch.nn as nn
import torch.nn.functional as F

def step_activation(x):
    return (x >= 0.0).float()

def sigmoid_activation(x):
    return torch.sigmoid(x)

def tanh_activation(x):
    return torch.tanh(x)

class TemporalRNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return torch.sigmoid(out)

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.drop(h_n.squeeze(0))
        out = self.fc(out)
        return torch.sigmoid(out)
