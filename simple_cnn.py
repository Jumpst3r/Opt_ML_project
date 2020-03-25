import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Simple CNN

class simpleCNN(nn.Module):
    def __init__(self, nbclasses):
        super(simpleCNN, self).__init__()
        self.L1 = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.L2 = nn.Sequential(
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.FC = nn.Sequential(
            nn.Linear(7 * 7 * 64,250),
            nn.ReLU(),
            nn.Linear(250,10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.L1(x)
        x = self.L2(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x