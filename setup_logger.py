# This file sets up logger utilities and fixes seeds to ensure reproducibility.
import os
import logging
import torch
import random
LOGGING_LEVEL = logging.INFO

# Setup logging
root = logging.getLogger()
root.setLevel(LOGGING_LEVEL)
format = "[%(asctime)s] [\u001b[32;1m %(levelname)s \u001b[0m] : %(message)s "
format_file = "[%(asctime)s] [%(levelname)s] : %(message)s "
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(format, date_format)
fileformatter = logging.Formatter(format_file, date_format)
file = logging.FileHandler("log.txt")
file.setLevel(level=logging.DEBUG)
file.setFormatter(fileformatter)
console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
console.setFormatter(formatter)
root.addHandler(console)
root.addHandler(file)
# Ensure reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)