
# This file implements the PSO algorithm to generate
# adverserial examples. The main differences with the version in the paper are:
# Tensor based batch processing instead of sequential processing. This helps with speed.
# The fitness function in the paper does not seem to make much sense. Changed it accordingly
# (see further down for more details)

import random
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Particle():
    # Param types:
    # @coordinates, @velocity, @best_local_arg: Tensor of shape [1,image_w*image_h*image_c]
    # @best_local_val, @fitness: Float
    # @swarm: Object of type Swarm()
    def __init__(self, coordinates, best_local_arg, best_local_val,velocity, swarm):
        self.coordinates = coordinates
        self.best_local_arg = best_local_arg
        self.best_local_val = best_local_val
        self.velocity = velocity
        self.parent_swarm = swarm
        self.fitness = -float("INF")
        self.swarm = swarm

    def update_fitness(self, c=2):
        improved_global = False
        adv_label = self.swarm.model(self.coordinates[None,:].view(1, self.swarm.channelNb, self.swarm.width, self.swarm.height))
        good_label = self.swarm.model(self.swarm.target_image)
        normdist = torch.norm(self.coordinates - self.swarm.target_image.view(1,self.swarm.width*self.swarm.height*self.swarm.channelNb), dim=1)
        conf = (adv_label[:,self.swarm.TRUECLASS].T)[0]
        fitness = -conf - c*normdist
        if fitness > self.fitness:
            self.best_local_arg = self.coordinates.clone()
            self.best_local_val = fitness
        if fitness > self.swarm.global_best_val:
            self.swarm.global_best_val = fitness
            self.swarm.best_particle_position = self.coordinates.clone()
            improved_global = True
        self.fitness = fitness
        return improved_global

class Swarm():
    def __init__(self, nb_particles, target_image, model, img_id=0, particle_inertia=0.3, coginitve_weight=2, social_weight=2, inf_norm=0.1):
        # Save image related atributes
        self.target_image = target_image
        self.width = self.target_image.shape[2]
        self.height = self.target_image.shape[3]
        self.channelNb = self.target_image.shape[1]

        # We know that the target image is correctly classified, retrieve the true label
        modelout = model(self.target_image)
        self.INITCONF, self.TRUECLASS = torch.max(modelout, 1)
        self.predicted_label = self.TRUECLASS
        
        self.img_id = img_id

        inital_particle_positions = [self.target_image.view(1,self.width*self.height*self.channelNb) for _ in range(nb_particles)]
        masks = [torch.randint_like(torch.Tensor(size=(1, self.width*self.height*self.channelNb)), 0, 10).to(device)==0 for _ in range(nb_particles)] # 1/10 chance of changing a pixel
        
        for p,m in zip(inital_particle_positions, masks): p[m] + 0.2 * (-2 * torch.rand(size=p[m].shape).to(device)+1)
        velocities = [torch.normal(0,1,size=(1, self.width*self.height*self.channelNb)).to(device) for _ in range(nb_particles)]

        self.particles = [Particle(p.clamp(-1,1),p.clone(),0,v,self) for p,v in zip(inital_particle_positions,velocities)]
        self.global_best_val = -float("INF")
        self.best_particle_position = None
        self.model = model
        self.model.eval()
        self.particle_inertia = particle_inertia
        self.cognitive_weight = coginitve_weight
        self.social_weight = social_weight
        
        for p in self.particles:
            p.update_fitness()
        
        self.inf_norm = inf_norm
        self.L2_norms = [] 
        self.conf = []
        self.diverged = True

    def step(self, epoch=0):
        best_epoch_fitness = 0
        for p in self.particles:
            r1 = torch.rand(size=(1, self.width*self.height*self.channelNb)).to(device)
            r2 = torch.rand(size=(1, self.width*self.height*self.channelNb)).to(device)
            p.velocity = (self.particle_inertia * p.velocity + self.cognitive_weight * r1 * (p.best_local_arg-p.coordinates) + self.social_weight * r2 * (self.best_particle_position-p.coordinates))
            p.coordinates += (p.velocity)

            # Make sure we didn't go over the inf_norm:
            p.coordinates = torch.where(p.coordinates > self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, p.coordinates)
            p.coordinates = torch.where(p.coordinates < self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, p.coordinates)
            p.coordinates.clamp(-1,1)
            p.update_fitness()
        if epoch % 10 == 0:
            adv_pred = (self.model(self.best_particle_position[None,:].view(1, self.channelNb,self.width, self.height)))
            cval , idx = torch.max(adv_pred, 1)
            self.predicted_label = idx
            # Peridically print stats
            print("[Img. Nr: {}][E:{}] \t\t L2: {:4f} \t predicted: {} \t should be: {}".format(self.img_id ,epoch, torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb)).item(),idx.item(), self.TRUECLASS.item()))
            # Store the values so that they can be plotted later on
            self.L2_norms.append(torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb)))
            self.conf.append(cval)
        self.diverged = (self.global_best_val == -float("INF"))