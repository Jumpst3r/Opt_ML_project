# This file creates the timing graphs

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
from plot_utils import plot_results
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
    if len(d) == 0:
        logger.error("confident_inputs/CIFAR_model/ empty, did you unzip data.7z ? (ref. README)")
        exit(1)
except FileNotFoundError:
    logger.error(" model saves not found, did you run model_trainer.py?")
    exit(1)
logger.info("loaded model(s) and data")


# How many images we want to test
IMG_NUM = 10
ITERATIONS = 30
MAX_PARTICLES = 50
"""
This small snippet of code benchmarks the time needed to perform 30 epochs with varying amounts of particles
on the batch and sequential PSO version.    
"""
time_batch_mean = []
time_seq_mean = []
x = []
for particle_count in range(10,MAX_PARTICLES):
    x.append(particle_count)
    logger.info(f"Benchmarking with {particle_count} particles")
    time_batch = []
   
    time_seq = []
    for img_id in range(IMG_NUM):
        model.eval()
        # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
        target_im = data[str(model)][img_id][None, :, :, :].to(device)

        # Create a new particle swarm
        swarm_1 = Swarm(particle_count, target_im, model, img_id)    
        swarm_2 = Slow_swarm(particle_count, target_im, model, img_id)    

        for i in range(ITERATIONS):
            start1 = timer()
            swarm_1.step(epoch=i)
            end1 = timer()

            start2 = timer()
            swarm_2.step(epoch=i)
            end2 = timer()
      
        time_batch.append(end1-start1)
        time_seq.append(end2-start2)
    time_batch_mean.append(sum(time_batch) / len(time_batch))
    time_seq_mean.append(sum(time_seq) / len(time_seq))
logger.info(f"mean time batch (10-{MAX_PARTICLES} particles): {time_batch_mean}")
logger.info(f"time sequential (10-{MAX_PARTICLES} particles): {time_seq_mean}")
plt.title("Mean required time for 30 fixed iterations (N=10)")
plt.scatter(x=x, y=time_batch_mean, label="Batch PSO implementation")
plt.scatter(x=x, y=time_seq_mean, label="Sequential PSO implementation")
plt.xlabel("Particle count")
plt.ylabel("Time (s)")
plt.legend()
plt.show()