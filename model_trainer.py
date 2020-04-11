import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torchvision import datasets, transforms
from models import *

# Select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Normalization functions
f_mnist = transforms.Compose([transforms.ToTensor(),
                              #transforms.Normalize((0.5,), (0.5,)),
                              ])

f_cifar = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                              ])

                
# Download data
training_set_MNIST = datasets.MNIST('.data/', download=True, train=True, transform=f_mnist)
trainloader_MNIST = torch.utils.data.DataLoader(training_set_MNIST, batch_size=64, shuffle=True)
test_set_MNIST = datasets.MNIST('.data/', download=True, train=False, transform=f_mnist)
testloader_MNIST = torch.utils.data.DataLoader(test_set_MNIST, batch_size=64, shuffle=True)
training_set_CIFAR10 = datasets.CIFAR10('.data/', download=True, train=True, transform=f_cifar)
trainloader_CIFAR10 = torch.utils.data.DataLoader(training_set_CIFAR10, batch_size=64, shuffle=True)
test_set_CIFAR10 = datasets.CIFAR10('.data/', download=True, train=False, transform=f_cifar)
testloader_CIFAR10 = torch.utils.data.DataLoader(test_set_CIFAR10, batch_size=64, shuffle=True)

loaders = [(trainloader_MNIST, testloader_MNIST), (trainloader_CIFAR10, testloader_CIFAR10)]

# Train a given model
def train(model, data, epochs):
    print("training <" + str(model) + '>...')
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    for e in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inp, target = data
            inp, target = inp.to(device), target.to(device)
            out = model(inp)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:
                print('[%d, %5d] loss: %f' %
                (e + 1, i + 1, running_loss / 100))
                running_loss = 0.0    

# Test a given model
# NOTE: The paper evaluates their PSO blackbox attack as follows:
# Given trained models they extract the first 1000 samples
# which are correctly classified. These samples are then modified
# to produce adverserial examples. (Note it would make no sense to
# modify samples which are already missclassified). So we need to
# extract 1000 correctly classified samples from each dataset
def test(model, testloader):
    print("testing <" + str(model) + '>...')
    correct = 0
    total = 0
    correct_samples = []
    with torch.no_grad():
        for data in testloader:
            inp, target  = data
            inp, target = inp.to(device), target.to(device)
            outputs = model(inp)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # This super inefficient loop is just to save 1000 correctly classified examples:
            for testin,(pred,tar) in zip(inp,zip(predicted, target)):
                if (pred == tar):
                    correct_samples.append((testin, tar.item()))
                if len(correct_samples) > 1000: break
        print(str(model) + ' test acc: %d %%' % (
            100 * correct / total))
    PATH = 'confident_input/' + str(model) + '/'
    print("saving 1000 correctly classified samples to " + PATH)
    for idx, e in enumerate(correct_samples):
        im, label = e
        # naming convention: im_ID_LABEL.data
        torch.save(im, PATH + 'im_' + str(idx) + '_' + str(label) + '.data')


# Define which models to train
models = [MNSIT_model(),CIFAR_model()]

# Train and save the models
for model, loader in zip(models, loaders):
    trainloader, testloader = loader
    model.to(device)
    train(model, trainloader, 3)
    model.eval()
    test(model, testloader)
    torch.save(model.state_dict(), 'models/' + str(model) + ".state")
