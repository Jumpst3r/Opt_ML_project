
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


# Remember: Swarm particles represent a slightly modified version of the original image.
# For efficiency we store all the particles in one big tensor of shape [Nb_particles, image_width * image_height * image_channel_count]
# Each particle (row) of that tensor has a number of values associated to it besides their coordinates. These include:
#  - swarm.velocities (particle velocities), again a tensor of shape [Nb_particles, image_width * image_height * image_channel_count]
#  - swarm.particle_fitness a tensor of shape [Nb_particles] which contains the "score" each particle currently achieves
#  - swarm.particle_best_pos (personal best known locations): a tensor of shape [Nb_particles, image_width * image_height * image_channel_count]
# The swarm as a whole keeps track of:
# - swarm.swarm_best_fitness: A value designating the best achieved fitness
# - best_particle_position: A tensor of shape [image_width * image_height * image_channel_count] containing the coordinates of the best known particle
# The swarm also contains a number of hyperpameters:
# - swarm.particle_inertia: particles inertia
# - swarm.coginitve_weight: How strongly particles are attracted to their own best known position
# - swarm.social_weight: How strongly particles are attracted to the best known global positon
# - swarm.inf_norm: Highest deviation factor for individual pixels. (If set to 0 we can't deviate from the original image)
class Swarm():
    def __init__(self, nb_particles, target_image, model, img_id=0, particle_inertia=0.3, coginitve_weight=2, social_weight=2, inf_norm=0.6):
        # Save image related atributes
        self.target_image = target_image.to(device)
        self.width = self.target_image.shape[2]
        self.height = self.target_image.shape[3]
        self.channelNb = self.target_image.shape[1]

        # We know that the target image is correctly classified, retrieve the true label
        modelout = model(self.target_image)
        self.INITCONF, self.TRUECLASS = torch.max(modelout, 1)
        self.predicted_label = self.TRUECLASS
        
        self.THRESH = []
        self.img_id = img_id

        # create a tensor of shape [Nb_particles, image_width * image_height * image_channel_count]
        self.particle_coordinates = (self.target_image + torch.zeros((nb_particles,self.channelNb,self.width,self.height)).to(device)).view(nb_particles, self.target_image.view(-1).shape[0])
        
        # Select some random indices from that tensor and change a few values with proba 1/10
        # This corresponds to creating a couple of copies of the original image with some changed pixels.
        # The 0.2 factor dictates how strongly the pixels are changed
        mask = torch.randint_like(self.particle_coordinates, 0, 10).to(device)==0 # 1/10 chance of changing a pixel
        self.particle_coordinates[mask] += 0.2 * (-2 * torch.rand(size=self.particle_coordinates[mask].shape)+1).to(device)
        self.particle_coordinates = self.particle_coordinates.clamp(-1,1)
        self.particle_best_pos = self.particle_coordinates.clone().to(device)
        
        # velocities are random at first (TODO: try out other intialization schemes)
        self.velocities = (torch.normal(0,1,size=(nb_particles, self.target_image.view(-1).shape[0]))).to(device)
        self.model = model.to(device)
        self.model.eval()
        self.particle_inertia = particle_inertia
        self.cognitive_weight = coginitve_weight
        self.social_weight = social_weight
        self.update_fitness()
        self.particle_fitness.to(device)
        self.swarm_best_fitness, idx = torch.max(self.particle_fitness,0)
        self.best_particle_position = self.particle_coordinates[idx]
        self.best_particle_position.to(device)
        self.inf_norm = inf_norm
        self.L2_norms = []
        self.conf = []
        self.diverged = True

    def step(self, epoch=0):
        # Update the particle positions in a vectorized manner
        # (instead of looping through all particles update the positions using tensor ops).
        # We have:

        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        self.velocities = (self.particle_inertia * self.velocities + self.cognitive_weight * r1 * (self.particle_best_pos-self.particle_coordinates) + self.social_weight * r2 * (self.best_particle_position-self.particle_coordinates))
        self.particle_coordinates += (self.velocities)

        # Make sure we didn't go over the inf_norm:
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
            self.diverged = False
            self.swarm_best_fitness = newfitness
            self.best_particle_position = self.particle_coordinates[newpos.item()]
            # Not really needed but we want to print some stats so we send the best particle through the model and look at
            # the predicated class / L2 norm.
            adv_pred = (self.model(self.best_particle_position[None,None,:].view(1, self.channelNb,self.width, self.height)))
            cval , idx = torch.max(adv_pred, 1)
            self.predicted_label = idx
            # Peridically print stats
            if epoch % 10 == 0:
                print("[Img. Nr: {}][E:{}] \t\t L2: {:4f} \t predicted: {} \t should be: {}".format(self.img_id ,epoch, torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb)).item(),idx.item(), self.TRUECLASS.item()))
            # Store the values so that they can be plotted later on
            self.L2_norms.append(torch.norm(self.best_particle_position - self.target_image.view(1,self.width*self.height*self.channelNb)))
            self.conf.append(cval)
        # And finally we update the particle's personal best values:
        # This is a boolean tensor of shape [Nbparticles] with True if the new fitness is better than the old one.
        # In this case we update the personal best location to the new particle location
        mask = (self.particle_fitness > tmp)
        self.particle_best_pos[mask] = self.particle_coordinates[mask]
    
    # This method updates the fitness values of all particles in the swarm
    # We define a fitness function which evaluates a particle's fitness using eq. 7
    # in the paper:
    # (We want to maximize this)
    # Note: The model outputs a confidence score for each class. Image we have the 10 MNIST classes
    # the model could output for a given sample: [1.3,5.4,0,5,2,1.23,2,1,1.2,4.2]
    # In this case the second entry (5.4) is highest so the model would predict the digit one.
    # So what we want to do is minimize the confidence of the correct class. (equiv: max -confidence)
    # In the paper they write that they want to maximize |p_0-p_1| where p_0 is the confidence in the correct class of the
    # non modified image and p_1 is the confidence in the correct class of the modified image. But p_0 never changes and in their
    # equation a very large p_1 would be good. But we want to minimize p_1.
    def update_fitness(self, c=1):
        # adv_label will be a tensor of shape [nb_particles, num_classes] and hold the prediction for every particle.
        # to pass our particles to the model we need to reshape our coordinate tensor of shape [nbParticles, nb_of_pixels] to [nbParticles,1,nb_of_pixels]
        # Note that by passing all the particles at once we take advantage of batch processing which greatly speeds things up.
        adv_label = self.model(self.particle_coordinates[:,None,:].view(self.particle_coordinates.shape[0], self.channelNb, self.width, self.height))
        good_label = self.model(self.target_image)
        normdist = torch.norm(self.particle_coordinates - self.target_image.view(1,self.width*self.height*self.channelNb), dim=1)
        conf = (adv_label[:,self.TRUECLASS].T)[0]
        # So we maximize this, which is equivalent to saying:
        # Minimize the confidence in the correct class
        # Penalize images which are far too different.
        # TODO: Insteas of using the L2 norm we could try out a distance metric which is better suited for image similarity:
        # (https://en.wikipedia.org/wiki/Structural_similarity)
        fitness = -conf - c*normdist
        self.particle_fitness = fitness