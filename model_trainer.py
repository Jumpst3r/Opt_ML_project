# This file trains a simple model defined in models.py on the CIFAR-10 dataset
#
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torchvision import datasets, transforms
from models import CIFAR_model
import torchvision.models as models
import glob, os
import setup_logger
import logging
logger = logging.getLogger()

# How many confident inputs to store.
NUMSAMPLES = 500

# Select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"using device {device}")

f_cifar = transforms.Compose([transforms.ToTensor()])

                
training_set_CIFAR10 = datasets.CIFAR10('.data/', download=True, train=True, transform=f_cifar)
trainloader_CIFAR10 = torch.utils.data.DataLoader(training_set_CIFAR10, batch_size=64, shuffle=True)
test_set_CIFAR10 = datasets.CIFAR10('.data/', download=True, train=False, transform=f_cifar)
testloader_CIFAR10 = torch.utils.data.DataLoader(test_set_CIFAR10, batch_size=64, shuffle=True)

loader = (trainloader_CIFAR10, testloader_CIFAR10)

# Remove previously generated confident outputs:

logger.info("Removing old files..")
files = glob.glob('confident_input/CIFAR_model/*.data')
for f in files:
    os.remove(f)


# Train a given model
def train(model, data, epochs):
    logger.info("training <" + str(model) + '>...')
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
                logger.info('Epoch %d, loss: %f' %
                (e + 1, running_loss / 100))
                running_loss = 0.0    


def test(model, testloader):
    logger.info("testing <" + str(model) + '>...')
    correct = 0
    total = 0
    correct_samples = []
    with torch.no_grad():
        for inp, target in testloader:
            inp, target = inp.to(device), target.to(device)
            outputs = model(inp)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # This super inefficient loop is just to save 1000 correctly classified examples:
            for testin,(pred,tar) in zip(inp,zip(predicted, target)):
                if (pred == tar):
                    correct_samples.append((testin, tar.item()))
                if len(correct_samples) > NUMSAMPLES: break
        logger.info(str(model) + ' test acc: %d %%' % (
            100 * correct / total))
    PATH = 'confident_input/' + str(model) + '/'
    logger.info(f"saving {NUMSAMPLES} correctly classified samples to " + PATH)
    for idx, e in enumerate(correct_samples):
        im, label = e
        # naming convention: im_ID_LABEL.data
        torch.save(im, PATH + 'im_' + str(idx) + '_' + str(label) + '.data')



model= CIFAR_model()
trainloader, testloader = loader
model.to(device)
train(model, trainloader, 10)
model.eval()
test(model, testloader)
logger.info("saving model state to models/")
torch.save(model.state_dict(), 'models/' + str(model) + ".state")


