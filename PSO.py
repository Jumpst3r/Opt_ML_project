import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from functools import reduce
from operator import mul
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Particle():
    def __init__(self, coordinates, pers_best):
        self.coordinates = coordinates
        self.pers_best = pers_best
        self.loss = float("INF")
        self.best_loss = float("INF")
        self.velocity = torch.normal(0,1,size=self.coordinates.shape).to(device)

    def update_position(self):
        self.coordinates = self.coordinates + self.velocity


class PSO(Optimizer):
    def __init__(self, model, cognitive_weight=1.9, social_weight=2, inertia=1e-8, nbparticles=50):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia = inertia
        self.params = model.parameters
        self.nbparticles = nbparticles
        self.best_loss = float("INF")
        # pytorch already initializes the model parameters so that's done.
        # Lets store a tensor of shape (nb_params) which will represent a particle in our search space
        self.first_particle = torch.tensor([]).float().to(self.device)
        self.particles = []
        for p in self.params():
            self.first_particle = torch.cat((self.first_particle.float(),p.data.view(-1).float()))
        # We need nbparticles of such candidates. Let's generate them by adding
        # normal noise to the values which pytorch created:
        self.best_loss_arg = self.first_particle
        for e in range(self.nbparticles):
            c = self.first_particle + torch.normal(0,0.4,size=self.first_particle.shape).float().to(self.device)
            self.particles.append(Particle(c, c))
        # Lets evaluate the loss for every particle and update the global best loss:
        
        # We avoid doing explicit initialisation here because we don't have access to the closure function.

        # We are now done with initialization, we have an initial best known global value and
        # nbparticles ready each containing random coordinates which consist of model parameter
        # candidates.

    def _update_parameters(self, particle: Particle):
        lim = 0 
        for p in self.params():
            s = p.data.shape
            nb_params = reduce(mul, s, 1)
            p.data = particle.coordinates[lim:lim+nb_params].reshape(s).float()
            lim = nb_params

    def step(self, closure):
        # sequentially:
        # update paramters
        #print("STEP-----------------------")
        for i,p in enumerate(self.particles):
            r1 = random.uniform(0,1)
            r2 = random.uniform(0,1)
            p.velocity = self.inertia * p.velocity + self.cognitive_weight * r1 * (p.coordinates - p.pers_best) + self.social_weight * r2 * (self.best_loss_arg-p.coordinates)
            p.update_position()
            self._update_parameters(p)
            p.loss = float(closure())
            #print("loss for p_{} = {}".format(i,p.loss))
            # update personal and global best values if we improved.
            if p.loss < self.best_loss:
                self.best_loss_arg = p.coordinates
                self.best_loss = p.loss
            if p.loss < p.best_loss:
                p.pers_best = p.coordinates
                p.best_loss = p.loss
        self.inertia = self.inertia-1e-3 if self.inertia-1e-3 > 0 else 1e-3 
