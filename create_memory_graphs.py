# This file generates the memory graphs

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
    if len(d) == 0:
        logger.error("confident_inputs/CIFAR_model/ empty, did you unzip data.7z ? (ref. README)")
        exit(1)
except FileNotFoundError:
    logger.error(" model saves not found, did you run model_trainer.py?")
    exit(1)
logger.info("loaded model(s) and data")


# How many images we want to test
IMG_NUM = 10
ITERATIONS = 100
MAX_PARTICLES = 50
avg_cuda_m = []
peak_cuda_m = []
x = []
for particle_count in range(10, MAX_PARTICLES):
    total_success = 0
    x.append(particle_count)
    avg_cuda = []
    peak_cuda = []
    logger.info(f"Benchmarking with {particle_count} particles")
    for img_id in range(IMG_NUM):
        model.eval()
        # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
        target_im = data[str(model)][img_id][None, :, :, :].to(device)
        # Create a new particle swarm
        swarm = Swarm(particle_count, target_im, model, img_id)    
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(ITERATIONS):
            swarm.step(epoch=i)
        end_event.record()
        torch.cuda.synchronize()
        avg_cuda.append(torch.cuda.memory_allocated())
        peak_cuda.append(torch.cuda.max_memory_allocated())
    avg_cuda_m.append( (sum(avg_cuda) / len(avg_cuda)) / 1e6 )
    peak_cuda_m.append((sum(peak_cuda) / len(peak_cuda)) / 1e6)

logger.info(f"batch PSO mean and peak GPU memory usage {avg_cuda_m}, {peak_cuda_m}")
plt.title("batch PSO mean and peak GPU memory usage (averaged over 10 image) / number of particles")
plt.scatter(x=x, y=avg_cuda_m, label="avg GPU memory consumption")
plt.scatter(x=x, y=peak_cuda_m, label="peak  GPU memory consumption")
plt.xlabel("Particle count")
plt.ylabel("Memory consumption in MB")
plt.legend()
plt.show()
