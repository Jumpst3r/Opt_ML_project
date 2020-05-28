# Analoguous to the benchmark_pso file but for the reference whitebox attacks
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
from plot_utils import plot_results_torchattacks
import torchattacks

MODEL_PATH = 'models/'
# confident inputs
DATA_PATH = 'confident_input/'
# results (uncomment line 98 to store image results)
RESULT_PATH = 'results/'


logger = logging.getLogger()


# remove old results (if any)
logger.info("Removing old files..")
files = glob.glob('results/*.png')
for f in files:
    os.remove(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




# Load the trained model and the correctly classified inputs:
model = CIFAR_model()

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
    logger.error("model saves not found, did you run model_trainer.py?")
    exit(1)
logger.info("loaded model and data")

CUDA_MODE = torch.cuda.is_available()

if not CUDA_MODE:
    logger.warn("CUDA not available. Run on a CUDA enabled platform (NVIDIA GPU with compute capability >= 3) to get memory usage and timing stats (this code makes use of CUDA events to accurately measure memory and timing). Press [ENTER] to continue anyways.")
    input()

# How many images we want to test
IMG_NUM = 500
model.eval()
attacks = [torchattacks.PGD(model), torchattacks.DeepFool(model), torchattacks.StepLL(model), torchattacks.BIM(model)]

for attack in attacks:
    time = []
    attack_l2 = []
    peak_cuda = []
    avg_cuda = []
    total_success = 0

    logger.info(f"Benchmarking {str(attack)} on {IMG_NUM} images")
    for img_id in range(IMG_NUM):
       
        # Load the image and reshape it to [NxCxWxH] (which is what the models expect)
        target_im = data[str(model)][img_id][None, :, :, :].to(device)
        _, TRUECLASS = torch.max(model(target_im), 1)
        logger.debug(f"[{str(attack)}] Prediction before attack is {TRUECLASS}")
        if CUDA_MODE: 
            torch.cuda.reset_peak_memory_stats(device=device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        image = attack(target_im, TRUECLASS)
        if CUDA_MODE: 
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)

        output = model(image)   
        _, pre = torch.max(output.data, 1)
    
        if TRUECLASS != pre:
            total_success += 1
            l2 = torch.norm(image.view(1,-1) - target_im.view(1,-1), dim=1).item()	
            # plot_results_torchattacks(target_im, pre, TRUECLASS, image, l2, str(attack), RESULT_PATH, img_id)
            attack_l2.append(l2)
            if CUDA_MODE: 
                avg_cuda.append(torch.cuda.memory_allocated())
                peak_cuda.append(torch.cuda.max_memory_allocated())
                time.append(elapsed_time_ms)
            logger.debug(f"[{str(attack)}] succeeded with L2: {l2}")
        else:
            logger.warning(f"\u001b[33m [Im {img_id}/{IMG_NUM}] attack failed on given image. \u001b[0m")
            pass
 
    # Print final statistics
    logger.info(f"\u001b[32;1m Done! Processed {IMG_NUM} images ({str(attack)}) with following stats: \u001b[0m")
    logger.info(f"success rate: {(total_success / IMG_NUM):.4f}")
    logger.info(f"mean L2:\t {(sum(attack_l2) / total_success):.4f}")
    if CUDA_MODE: 
        logger.info("mean processing time:\t\t {:.4f}".format((sum(time) / total_success)/ 1000))
        logger.info("mean peak GPU memory :\t\t {:.4f} MB".format((sum(peak_cuda) / total_success)/1e6))
        logger.info("avg GPU memory :\t\t {:.4f} MB".format((sum(avg_cuda) / total_success)/1e6))
