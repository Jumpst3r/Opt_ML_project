# Opt_ML_project
A repository for our project for the optimization for machine learning course

The overall goal is: Implement the PSO adverserial attack from:
https://arxiv.org/abs/1909.07490

## Files and structure

So the process is broken down into several parts

### Training reference models

The file `models.py` contains pytorch implementations of the models used with some popular datasets to benchmark the implementations. These models were ported from tf/keras to pytorch from https://github.com/huanzhang12/ZOO-Attack

These models are trained in `model_trainer.py`

Once the models are trained, they are saved under `models`. We also save 1000 correctly classified images to `confident_input`.
(These are needed to benchmark the adverserial attacks, it would
make no sense to craft an adverserial example starting from an image which is already missclassified.)

### Loading the models, data and performing the attack

The file `main.py` loads the saved models from `models/` and the
reference input data from `confident_input/`.

TODO: Maybe add the imagenet model and then start with the PSO attack implementation.

### Whitebox Baselines:

Borrow codes from: https://github.com/Harry24k/adversarial-attacks-pytorch#Demos
The notebook I use is in: "White Box Attack Cifar-10.ipynb"
There are several basesline and their computation time and L2 loss per iter. Also, I provide the correct num.
To run the notebook, first run: 
'''
pip install torchattacks
'''