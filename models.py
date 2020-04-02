import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Same architecture used in the ZOO paper for the MNIST dataset
# (ref. https://github.com/huanzhang12/ZOO-Attack/blob/master/setup_mnist.py)
# MNIST tensor dimensions: [c,w,h] = [1,28,28], 10 labels
class MNSIT_model(nn.Module):
    def __init__(self):
        super(MNSIT_model, self).__init__()
        self.convblock = nn.Sequential(
            # Block 1
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Block 2
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fcblock = nn.Sequential(
            # FC Block
            nn.Linear(64*4*4, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.convblock(x)
        x = x.view(-1, 64*4*4)
        x = self.fcblock(x)
        return x
    
    def __str__(self):
        return "MNIST_model"

# Same architecture used in the ZOO paper for the CIFAR 10 dataset
# (ref. https://github.com/huanzhang12/ZOO-Attack/blob/master/setup_cifar.py)
# MNIST tensor dimensions: [c,w,h] = [3,32,32], 10 labels
class CIFAR_model(nn.Module):
    def __init__(self):
        super(CIFAR_model, self).__init__()
        self.convblock = nn.Sequential(
            # Block 1
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # Block 2
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fcblock = nn.Sequential(
            # FC Block
            nn.Linear(128*5*5, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.convblock(x)
        x = x.view(-1, 128*5*5)
        x = self.fcblock(x)
        return x
    
    def __str__(self):
        return "CIFAR_model"