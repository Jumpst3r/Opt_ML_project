# This file assumes that the models and data have been trained and
#  generated using 'model_trainer.py'
# It implements the PSO blackbox attacks based on
# https://arxiv.org/pdf/1909.07490.pdf
# The paths should work as defined
MODEL_PATH = 'models/'
DATA_PATH = 'confident_input/'

import matplotlib.pyplot as plt
import random
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import MNSIT_model, CIFAR_model
CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
torch.set_grad_enabled(False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# If set to true only load MNIST cof inputs / model (CIFAR takes a lot longer this allows faster prototyping)
#MNIST_ONLY = True
CIFAR_ONLY = True

# Load the trained models and the correctly classified inputs:
models = [MNSIT_model(), CIFAR_model()] if not CIFAR_ONLY else [CIFAR_model()]
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
# At this point we can start implementing the
# PSO attack from the paper (yay).
# TODO: Implement the PSO attack from the paper

# Load the first image and reshape it to [NxCxWxH] (which is what the models expect)
target_im = data[str(models[0])][0][None,:,:,:].to(device)
model = models[0]
model.eval()
INIT_CONF_, TRUECLASS = torch.max(model(target_im), 1)
OUTPUT = model(target_im)
SAVEFIGS = False
L2_norms = []
conf = []
plt.axis('off')
plt.subplot(2, 2, 1)
plt.title("Prediction: " + CIFAR_CLASSES[TRUECLASS.item()])
fig1 = plt.imshow(target_im[0].permute(1,2,0).cpu())
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
print("TRUE CLASS:", TRUECLASS)
class Swarm():
    def __init__(self, nb_particles, target_image, model, particle_inertia=0.004, coginitve_weight=1.6, social_weight=0.4, inf_norm=1):
        # We store all particle parameters in big tensors for performance reasons.
        self.target_image = target_image.to(device)
        self.width = target_im.shape[2]
        self.height = target_im.shape[3]
        self.channelNb = target_im.shape[1]
        self.minval = torch.min(self.target_image)
        self.maxval = torch.max(self.target_image)
        self.particle_coordinates = (self.target_image + torch.normal(0,0.2,size=(nb_particles,self.channelNb,self.width,self.height)).to(device)).view(nb_particles, target_image.view(-1).shape[0]).clamp(0-1,1)
        self.particle_best_pos = self.particle_coordinates.clone().to(device)
        
        self.velocities = (torch.normal(0,1,size=(nb_particles, target_image.view(-1).shape[0]))).to(device)
        self.model = model.to(device)
        self.particle_inertia = particle_inertia
        self.cognitive_weight = coginitve_weight
        self.social_weight = social_weight
        self.update_fitness()
        self.particle_fitness.to(device)
        self.swarm_best_fitness, idx = torch.max(self.particle_fitness,0)
        self.best_particle_position = self.particle_coordinates[idx]
        self.best_particle_position.to(device)
        self.inf_norm = inf_norm

    def step(self, epoch=0):
        # Update the particle positions in a vectorized manner
        # (instead of looping through all particles update the positions using matrix ops).
        # We have:
        # self.particle_coordinates: Tensor of shape [Nb_particles, Nb_pixels] which represents all of our candidates
        # self.fitness(): Returns a Tensor of shape [Nb_particles] containing the fitness of each particle
        # self.swarm_best_fitness: Best swarm fitness *value*
        # self.best_particle_position: Tensor of shape [1, Nb_pixels] which represents the best (global) candidate coordinate
        # self.velocities: Tensor of shape [Nb_particles, Nb_pixels] which represents particle per coordinate velocities

        # This implements the "conventional PSO" described in Eq 1 of the paprt

        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        self.velocities = (self.particle_inertia * self.velocities + self.cognitive_weight * r1 * (self.particle_best_pos-self.particle_coordinates) + self.social_weight * r2 * (self.best_particle_position-self.particle_coordinates))
        self.particle_coordinates += (self.velocities)
        self.particle_coordinates = torch.where(self.particle_coordinates > self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.particle_coordinates)
        self.particle_coordinates = torch.where(self.particle_coordinates < self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.particle_coordinates)
        self.particle_coordinates.clamp(-1,1)
        # Now we reavaluate the fitness at their new positions:
        # (we need a copy of the old fitness vals to see if we improved the particles pers. best):
        tmp = self.particle_fitness.clone()
        self.update_fitness()
         # Now let's get the best swarm fitness and position:
        newfitness , newpos= torch.max(self.particle_fitness,0)
        if newfitness > self.swarm_best_fitness:
            self.swarm_best_fitness = newfitness
            self.best_particle_position = self.particle_coordinates[newpos.item()]
            print("[E:{}] new optimum ({}) found".format(epoch, self.swarm_best_fitness))
            adv_pred = (model(self.best_particle_position[None,None,:].view(1, self.channelNb,self.width, self.height)))
            cval , idx = torch.max(adv_pred, 1)
            self.predicted_label = idx
            print("(L2 norm: {}".format(torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb))))
            print("model predicted label {}, should be label {}".format(idx.item(), TRUECLASS.item()) )
            print("confidence: ", adv_pred)
            L2_norms.append(torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb)))
            conf.append(cval)
        # And finally we update the particle's personal best values:
        # This is a boolean tensor of shape [Nbparticles] with True if the new fitness is better than the old one.
        # In this case we update the personal best location to the new particle location
        mask = (self.particle_fitness > tmp)
        self.particle_best_pos[mask] = self.particle_coordinates[mask]
    
    # This method updates the fitness values of all particles in the swarm
    # We define a fitness function which evaluates a particle's fitness using eq. 7
    # in the paper:
    # (We want to maximize this)
    def update_fitness(self, c=0.02222):
        # adv_label will be a tensor of shape [nb_particles, num_classes] and hold the prediction for every particle.
        # to pass our particles to the model we need to reshape our coordinate tensor of shape [nbParticles, nb_of_pixels] to [nbParticles,1,nb_of_pixels]
        # Note that by passing all the particles at once we take advantage of batch processing which greatly speeds things up.
        adv_label = model(self.particle_coordinates[:,None,:].view(self.particle_coordinates.shape[0], self.channelNb, self.width, self.height))
        good_label = model(self.target_image)
        normdist = torch.norm(self.particle_coordinates - self.target_image.view(1,self.width*self.height*self.channelNb), dim=1)
        fitness = c*torch.pow(normdist,2)
        conf = (adv_label[:,TRUECLASS].T)[0]
        self.particle_fitness = -conf-fitness
swarm = Swarm(1000, target_im, model)

for i in range(100):
    swarm.step(i)

plt.subplot(2, 2, 2)
plt.title("Prediction: " + CIFAR_CLASSES[swarm.predicted_label.item()])
fig2 = plt.imshow(swarm.best_particle_position.view(swarm.channelNb, swarm.width,swarm.height).permute(1,2,0).cpu())
fig2.axes.get_xaxis().set_visible(False)
fig2.axes.get_yaxis().set_visible(False)

plt.axis('on')

plt.subplot(2, 2, 3)
plt.title("L2 norm / epochs")

plt.plot(L2_norms)

plt.subplot(2, 2, 4)
plt.title("confidence in correct class / epoch")
plt.plot(conf)

plt.savefig("cifar_adv.png")
