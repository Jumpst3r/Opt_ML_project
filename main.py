# This file assumes that the models and data have been trained and
#  generated using 'model_trainer.py'
# It implements the PSO blackbox attacks based on
# https://arxiv.org/pdf/1909.07490.pdf with some differences 
# See PSO.py for details

# The paths should work as defined

# models are save here:
MODEL_PATH = 'models/'
# confident inputs
DATA_PATH = 'confident_input/'
# results
RESULT_PATH = 'results/'


import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import MNSIT_model, CIFAR_model
from PSO import Swarm

# Reproducible results:
import random
random.seed(0)
torch.manual_seed(0)

# remove old results (if any)
print("[+] Removing old files..")
files = glob.glob('results/*.png')
for f in files:
    os.remove(f)

CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# change to MNIST for the MNIST dataset
mode = "CIFAR"

# Load the trained models and the correctly classified inputs:
models = [MNSIT_model(), CIFAR_model()]

if mode == "CIFAR": models = [models[1]]
if mode == "MNIST": models = [models[0]]

data = {}
try:
    for m in models:
        m.load_state_dict(torch.load(MODEL_PATH + str(m) + ".state"))
        m.to(device)
        d = []
        for filename in glob.glob(DATA_PATH + str(m) + '/' + '*.data'):
            d.append(torch.load(filename))
        data[str(m)] = d
except FileNotFoundError:
    print("[!] model saves not found, did you run model_trainer.py?")
print("[+] loaded model(s) and data")


# Now we have the model and inputs. (We can load the data from data[<model_ID>])

# Some counters to keep track of performance
total_success = 0
L2 = 0
# How many images we want to test (out of the 1K). 
IMG_NUM= 40
diverged_count = 0
for img_id in range(IMG_NUM):
    model = models[0]
    model.eval()
    # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
    target_im = data[str(models[0])][img_id][None,:,:,:].to(device)

    # Create a new particle swarm
    swarm = Swarm(100, target_im, model, img_id)
    
    # plot initial image / prediction
    plt.axis('off')
    plt.subplot(2, 2, 1)
    if mode == "CIFAR": 
        plt.title("Prediction: " + CIFAR_CLASSES[swarm.TRUECLASS.item()])
        img = (target_im[0].permute(1,2,0).cpu())
        fig1 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)))
    if mode == "MNIST": 
        plt.title("Prediction: " + str(swarm.TRUECLASS.item()))
        img = target_im[0][0].cpu()
        fig1 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)), cmap="gray")

    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)

    #### main optimization loop #######

    for i in range(200):
        swarm.step(i)

    ###################################

    # Sometimes the PSO diverges
    if swarm.diverged: 
        print("[Img. Nr: {}] PSO divereged !".format(img_id))
        diverged_count += 1
    print("-----------------------------")

    # If the best candidate of the swarm has a different label than the true class we win.
    if (swarm.predicted_label != swarm.TRUECLASS): total_success += 1
    if len(swarm.L2_norms) != 0:
        L2 += swarm.L2_norms[-1]
   
    # Plot the generated image with the L2 loss and confidence scores
    plt.subplot(2, 2, 2)
    if mode == "CIFAR": 
        plt.title("Prediction: " + CIFAR_CLASSES[swarm.predicted_label.item()])
        img = swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height).permute(1,2,0).cpu()
        fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))
    if mode == "MNIST":
        plt.title("Prediction: " + str(swarm.predicted_label.item()))
        img = swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height)[0].cpu()
        fig2 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)), cmap="gray")
    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    plt.axis('on')

    plt.subplot(2, 2, 3)
    plt.title("L2 norm / epochs")

    plt.plot(swarm.L2_norms)

    plt.subplot(2, 2, 4)
    plt.title("confidence in correct class / epoch")
    plt.plot(swarm.conf)

    # Save the image to the results/ folder
    plt.savefig(RESULT_PATH + "result_{}.png".format(img_id))
    plt.clf()

# Print final statistics
print("--------------------------------------------------")
print("success rate: {}".format(total_success / (IMG_NUM-diverged_count)))
print("mean L2: {:4f}".format(L2 / (IMG_NUM-diverged_count)))