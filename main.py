# This file assumes that the models and data have been trained and
#  generated using 'model_trainer.py'
# It implements the PSO blackbox attacks based on
# https://arxiv.org/pdf/1909.07490.pdf
# The paths should work as defined
MODEL_PATH = 'models/'
DATA_PATH = 'confident_input/'

import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import MNSIT_model, CIFAR_model

# Load the trained models and the correctly classified inputs:
models = [MNSIT_model(), CIFAR_model()]
data = {}
try:
    for m in models:
        m.load_state_dict(torch.load(MODEL_PATH + str(m) + ".state"))
        d = []
        for filename in glob.glob(DATA_PATH + str(m) + '/' + '*.data'):
            d.append(torch.load(filename))
        data[str(m)] = d
except FileNotFoundError:
    print("[!] model saves not found, did you run model_trainer.py?")

# Now we have the model and inputs. (We can load the data from data[<model_ID>])
# At this point we can start implementing the
# PSO attack from the paper (yay).
# TODO: Implement the PSO attack from the paper