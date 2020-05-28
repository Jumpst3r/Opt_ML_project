# This file assumes that the models and data have been trained and
# generated using 'model_trainer.py' or that the compressed archive in confident_inputs/CIFAR_model has been decompressed!
# The file also assumes that it is run on a CUDA enabled platform.

import random
from timeit import default_timer as timer
import logging
import setup_logger
from PSO import Swarm as Batch_Swarm
from models import CIFAR_model
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import glob
import matplotlib.pyplot as plt
from plot_utils import plot_results_pso
import torchattacks
from PSO_sequential import Swarm as Seq_Swarm
torch.set_grad_enabled(False)

#### Replace Batch_Swarm with Seq_Swarm to benchmark the sequential PSO
# Swarm = Seq_Swarm
Swarm = Batch_Swarm

MODEL_PATH = 'models/'
# confident inputs
DATA_PATH = 'confident_input/'
# results (only works if line 118 is uncommented)
RESULT_PATH = 'results/'


logger = logging.getLogger()


# remove old results (if any)
logger.info("Removing old files..")
files = glob.glob('results/*.png')
for f in files:
    os.remove(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_MODE = torch.cuda.is_available()

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


if not CUDA_MODE:
    logger.warn("CUDA not available. Run on a CUDA enabled platform (NVIDIA GPU with compute capability >= 3) to get memory usage and timing stats (this code makes use of CUDA events to accurately measure memory and timing). Press [ENTER] to continue anyways.")
    input()

# Some counters to keep track of performance
total_success = 0
L2 = 0
L2_reduced = 0
peak_cuda = []
avg_cuda = []
time = []

# How many images we want to test
IMG_NUM = 500
ITERATIONS = 30

for img_id in range(IMG_NUM):
    model.eval()
    # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
    target_im = data[str(model)][img_id][None, :, :, :].to(device)

    # Create a new particle swarm
    swarm = Swarm(30, target_im, model, img_id)    
    success = False

    if CUDA_MODE: 
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    #### main optimization loop #######

    for i in range(ITERATIONS):
        success = False
        if swarm.step(epoch=i):
            success = True
            break
    ###################################

    if success:
        swarm.reduce()
        if CUDA_MODE: 
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_cuda.append(torch.cuda.memory_allocated())
            peak_cuda.append(torch.cuda.max_memory_allocated())
            time.append(elapsed_time_ms)
        total_success += 1
        L2_reduced += swarm.get_l2()
        # plot_results_pso(swarm, RESULT_PATH, img_id)
    else:
        logger.warning(f"\u001b[33m [Im {img_id}/{IMG_NUM}] PSO attack failed on given image. \u001b[0m")

# Print final statistics
logger.info("\u001b[32;1m Done! Processed {} images ({}, {} iterations) with following stats: \u001b[0m".format(IMG_NUM, str(swarm), ITERATIONS))
logger.info("success rate: {:.4f}".format(total_success / IMG_NUM))
logger.info("mean L2:\t {:.4f}".format(L2_reduced / total_success))
if CUDA_MODE: 
    logger.info("mean processing time (PSO attack):\t\t {:.4f}".format((sum(time) / total_success)/ 1000))
    logger.info("mean peak CUDA memory usage (PSO attack):\t\t {:.4f} MB".format((sum(peak_cuda) / total_success)/1e6))
    logger.info("mean avg CUDA memory usage (PSO attack):\t\t {:.4f} MB".format((sum(avg_cuda) / total_success)/1e6))
