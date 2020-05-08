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
import setup_logger
import logging

logger = logging.getLogger()

# Reproducible results:
import random
random.seed(0)
torch.manual_seed(0)

# remove old results (if any)
logger.info("Removing old files..")
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
    logger.error(" model saves not found, did you run model_trainer.py?")
    exit(1)
logger.info("loaded model(s) and data")

plt.rcParams["axes.titlesize"] = 8

# Now we have the model and inputs. (We can load the data from data[<model_ID>])

# Some counters to keep track of performance
total_success = 0
L2 = 0
L2_reduced = 0
# How many images we want to test (out of the 1K). 
IMG_NUM= 40
for img_id in range(IMG_NUM):
    model = models[0]
    model.eval()
    # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
    target_im = data[str(models[0])][img_id][None,:,:,:].to(device)

    # Create a new particle swarm
    swarm = Swarm(100, target_im, model, img_id)
    
    # plot initial image / prediction
    plt.axis('off')
    plt.subplot(1, 3, 1)
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

    for i in range(100):
        swarm.step(i)

    ###################################

    swarm.reduce()

    # If the best candidate of the swarm has a different label than the true class we win.
    if (swarm.get_predicted_label() != swarm.TRUECLASS): total_success += 1
			
    
    L2 += swarm.get_l2(swarm.before_reduce)
    L2_reduced += swarm.get_l2()
   
    # Plot the generated image with the L2 loss and confidence scores
    plt.subplot(1, 3, 2)
    if mode == "CIFAR": 
        plt.title("Prediction: " + CIFAR_CLASSES[swarm.predicted_label.item()] + "\n(before reduction L2={:.2f})".format(swarm.get_l2(swarm.before_reduce)))
        img = swarm.before_reduce.view(swarm.channelNb, swarm.width,swarm.height).permute(1,2,0).cpu()
        fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))
    if mode == "MNIST":
        plt.title("Prediction: " + str(swarm.predicted_label.item()) + "\n(before reduction L2={:2f})".format(swarm.get_l2(swarm.before_reduce)))
        img = swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height)[0].cpu()
        fig2 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)), cmap="gray")

    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    plt.subplot(1, 3, 3)
    if mode == "CIFAR": 
        plt.title("Prediction: " + CIFAR_CLASSES[swarm.predicted_label.item()]+ "\n(after reduction L2={:.2f})".format(swarm.get_l2()))
        img = swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height).permute(1,2,0).cpu()
        fig2 = plt.imshow((img-torch.min(img))/(torch.max(img)-torch.min(img)))
    if mode == "MNIST":
        plt.title("Prediction: " + str(swarm.predicted_label.item()) + "\n(after reduction L2={:.2f})".format(swarm.get_l2()))
        img = swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height)[0].cpu()
        fig2 = plt.imshow((img-torch.min(img)) / (torch.max(img)-torch.min(img)), cmap="gray")
    

    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)

    # Save the image to the results/ folder
    plt.savefig(RESULT_PATH + "result_{}.png".format(img_id))
    plt.clf()

# Print final statistics
logger.info("\u001b[32;1m Done! Processed {} images with following stats: \u001b[0m".format(IMG_NUM))
logger.info("success rate: {}".format(total_success / IMG_NUM))
logger.info("mean L2 (before reduction):\t {:.4f}".format(L2 / IMG_NUM))
logger.info("mean L2 (after reduction):\t {:.4f}".format(L2_reduced / IMG_NUM))
