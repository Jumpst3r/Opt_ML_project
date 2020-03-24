import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np
import math
from functools import reduce
from operator import mul
import random


class Particle():
    def __init__(self, coordinates, pers_best):
        self.coordinates = coordinates
        self.pers_best = pers_best
        self.loss = float("INF")
        self.best_loss = float("INF")
        self.velocity = -2 * torch.rand(self.coordinates.shape) + 1

    def update_position(self):
        self.coordinates = self.coordinates + self.velocity


class PSO(Optimizer):
    def __init__(self, model, closure, cognitive_weight=0.5, social_weight=1, inertia=0.3, nbparticles=10):
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia = inertia
        self.params = model.parameters
        self.nbparticles = nbparticles
        self.best_loss = float("INF")
        # pytorch already initializes the model parameters so that's done.
        # Lets store a tensor of shape (nb_params) which will represent a particle in our search space
        self.first_particle = torch.tensor([]).float()
        self.particles = []
        for p in self.params():
            self.first_particle = torch.cat((self.first_particle.float(),p.data.view(-1).float()))
        print("******************")
        # We need nbparticles of such candidates. Let's generate them by adding
        # normal noise to the values which pytorch created:
        for e in range(self.nbparticles):
            c = self.first_particle + torch.normal(0,3,size=self.first_particle.shape).float()
            self.particles.append(Particle(c, c))
        # Lets evaluate the loss for every particle and update the global best loss:
        
        for p in self.particles:
            self._update_parameters(p)
            loss = float(closure())
            if loss < self.best_loss:
                self.best_loss_arg = p.coordinates
                self.best_loss = loss

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