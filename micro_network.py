import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torchvision import datasets, transforms
from PSO import PSO
from simple_cnn import simpleCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

f = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
                


training_set = datasets.MNIST('.data/', download=True, train=True, transform=f)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True)


model = simpleCNN(10)
model.to(device)

def train_sgd(model, data, epochs):
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), 0.1)
    for e in range(epochs):
        epoch_best_loss = float('INF')
        for inp,target in iter(data):
            inp, target = inp.to(device), target.to(device)
            out = model(inp)
            loss = criterion(out, target)
            epoch_best_loss = min(epoch_best_loss, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch best loss: \t {}".format(loss.item()))

def train_pso(model, data, epochs):
    criterion = F.cross_entropy
    optimizer = PSO(model)
    for e in range(epochs):
        best_loss = float('INF')
        for inp,target in iter(data):
            inp, target = inp.to(device), target.to(device)
            def _closure():
                out = model(inp)
                return criterion(out, target)
            out = model(inp)
            optimizer.step(_closure)
            loss = optimizer.best_loss
            if (loss < best_loss):
                print("new best loss: \t {}".format(loss))



#train_pso(model, training_in, training_out, 10, 2)
train_pso(model, trainloader, 5)