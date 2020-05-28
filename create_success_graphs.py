# This file creates the success graphs

import random
from timeit import default_timer as timer
import logging
import setup_logger
from PSO import Swarm
from models import CIFAR_model
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import glob
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import torchattacks
from PSO_sequential import Swarm as Slow_swarm
torch.set_grad_enabled(False)

MODEL_PATH = 'models/'
# confident inputs
DATA_PATH = 'confident_input/'
# results
RESULT_PATH = 'results/'


logger = logging.getLogger()

# remove old results (if any)
logger.info("Removing old files..")
files = glob.glob('results/*.png')
for f in files:
    os.remove(f)

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat",
                 "deer", "dog", "frog", "horse", "ship", "truck"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_MODE = torch.cuda.is_available()
# change to MNIST for the MNIST dataset
mode = "CIFAR"

# Load the trained models and the correctly classified inputs:
model =  CIFAR_model()

data = {}
try:
    model.load_state_dict(torch.load(MODEL_PATH + str(model) + ".state", map_location=lambda storage, loc: storage))
    model.to(device)
    d = []
    for filename in glob.glob(DATA_PATH + str(model) + '/' + '*.data'):
        d.append(torch.load(filename, map_location=lambda storage, loc: storage))
    data[str(model)] = d
except FileNotFoundError:
    logger.error(" model saves not found, did you run model_trainer.py?")
    exit(1)
logger.info("loaded model(s) and data")


# How many images we want to test
IMG_NUM = 100
ITERATIONS = 50
MAX_PARTICLES = 300
success_rates = []
x = []
for particle_count in range(10, MAX_PARTICLES):
    total_success = 0
    x.append(particle_count)
    continue
    logger.info(f"Benchmarking with {particle_count} particles")
    for img_id in range(IMG_NUM):
        model.eval()
        # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
        target_im = data[str(model)][img_id][None, :, :, :].to(device)
        # Create a new particle swarm
        swarm = Swarm(particle_count, target_im, model, img_id)    
        success = False
        for i in range(ITERATIONS):
            success = False
            if swarm.step(epoch=i):
                success = True
                total_success += 1
                break
    success_rates.append(total_success / IMG_NUM)        
    
y = success_rates
logger.info(f"batch PSO success rates (10-{MAX_PARTICLES} particles): {success_rates}")
plt.title("Batch PSO sucess rate on 50 image / number of particles")
plt.scatter(x=x, y=y, marker="x", c="gray")
plt.xlabel("Particle count")
plt.ylabel("Success rate")
plt.legend()
plt.show()
