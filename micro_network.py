import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def gen_data(nb):
    input_class1 = torch.normal(0,1 , size=(nb,2))
    input_class2 = torch.normal(3,1 , size=(nb,2))
    input_class1 = torch.cat((input_class1, torch.ones((nb,1))), dim=1)
    input_class2 = torch.cat((input_class2, torch.zeros((nb,1))), dim=1)
    # concat both classes:
    data = torch.cat((input_class1,input_class2), dim=0)
    #print(data.shape)
    
    idx = torch.randperm(data.shape[0])
    data = data[idx]
    plt.scatter(data[:,0], data[:,1], c=data[:,2])
    plt.show()
    return data[:,0:2], data[:,2].long()

training_in, training_out = gen_data(100)
training_in.sub_(training_in.mean()).div_(training_in.std())
test_in, test_out = gen_data(50)

model = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2),
    nn.ReLU()
)

def train(model, data_in, data_expected, epochs, batch_size):
    criterion = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), 0.1)
    for e in range(epochs):
        for b in range(0, training_in.shape[0], batch_size):
            out = model(training_in[b:b+batch_size])
            loss = criterion(out, data_expected[b:b+batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss: {}".format(loss.item()))

train(model, training_in, training_out, 25, 100)