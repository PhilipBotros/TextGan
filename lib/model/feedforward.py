import torch
from torch import nn

class FeedForward(nn.Module):
    """ Simple feedforward (MLP) for alignment scores """
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        out = self.fc2(h1)
        return out