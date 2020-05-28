
# This file implements the PSO algorithm to generate
# adverserial examples. The main differences with the version in the paper are:
# Tensor based batch processing instead of sequential processing. This helps with speed.
# The fitness function in the paper does not seem to make much sense. Changed it accordingly
# (see further down for more details)

import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger()

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
    def __str__(self):
        return f"batch PSO (nb_particles={self.nb_particles}, particle_inertia={self.particle_inertia}, cognitive_weight={self.cognitive_weight}, social_weight={self.social_weight}, inf_norm={self.inf_norm} )"
    def __init__(self, nb_particles, target_image, model, img_id=0, particle_inertia=0.3, coginitve_weight=2, social_weight=2, inf_norm=0.1):
        """Initializes the particle swarm

        Args:
            nb_particles ([Integer]): Number of particles (candidate images in our case)
            target_image ([torch.Tensor]): The originaly correctly classified image we want to attack
            model (torch.model): The deep model we want to fool which orginally correctly classifies the target image.
            img_id (int, optional): Numerical ID of the current image. Defaults to 0.
            particle_inertia (float, optional): Inertia hyperparameter of the swarm. Defaults to 0.3.
            coginitve_weight (int, optional): cognitive hyperparameter of the swarm. Describes attraction to local best. Defaults to 2.
            social_weight (int, optional): social hyperparamter of the swarm. Describes attraction to global best. Defaults to 2.
            inf_norm (float, optional): [description]. Defaults to 0.1.
        """
        
        # Save image related atributes
        self.target_image = target_image.to(device)
        self.width = self.target_image.shape[2]
        self.height = self.target_image.shape[3]
        self.channelNb = self.target_image.shape[1]
        self.nb_particles = nb_particles

        # We know that the target image is correctly classified, retrieve the true label
        modelout = model(self.target_image)
        self.INITCONF, self.TRUECLASS = torch.max(modelout, 1)
        self.predicted_label = self.TRUECLASS
        
        self.THRESH = []
        self.img_id = img_id

        # create a tensor of shape [Nb_particles, image_width * image_height * image_channel_count]
        self.particle_coordinates = (self.target_image.clone() + torch.zeros((nb_particles,self.channelNb,self.width,self.height)).to(device)).view(nb_particles, self.target_image.view(-1).shape[0])
        
        # Select some random indices from that tensor and change a few values with proba 1/10
        # This corresponds to creating a couple of copies of the original image with some changed pixels.
        # The 0.2 factor dictates how strongly the pixels are changed
        mask = torch.randint_like(self.particle_coordinates, 0, 3).to(device)==0 # 1/10 chance of changing a pixel
        self.particle_coordinates[mask] += 0.3 * (-2 * torch.rand(size=self.particle_coordinates[mask].shape, device=device)+1)
        self.particle_coordinates = self.particle_coordinates.clamp(-1,1)
        self.particle_best_pos = self.particle_coordinates.clone().to(device)
        
        # velocities are random at first (TODO: try out other intialization schemes)
        self.velocities = torch.normal(0,1,size=(nb_particles, self.target_image.view(-1).shape[0]), device=device)
        self.model = model.to(device)
        self.model.eval()
        self.particle_inertia = particle_inertia
        self.cognitive_weight = coginitve_weight
        self.social_weight = social_weight
        self.update_fitness()
        self.swarm_best_fitness, idx = torch.max(self.particle_fitness,0)
        self.best_particle_position = self.particle_coordinates[idx]
        self.inf_norm = inf_norm
        self.L2_norms = []
        self.reduced_L2_norms = []
        self.conf = []
        self.diverged = True
        self.before_reduce = self.best_particle_position

    def update_predicted_label(self):
        """Updates the predicted label corresponding to the best known particle
        """
        adv_pred = (self.model(self.best_particle_position[None,None,:].view(1,self.channelNb,self.width,self.height)))
        _ , idx = torch.max(adv_pred, 1)
        self.predicted_label = idx


    def step(self, epoch=0):
        """Performs one optimization step, consisting of particle position update followed by new position fitness updates.

        Args:
            epoch (int, optional): [Epoch identifier]. Defaults to 0.

        Returns:
            [Boolean]: [True if we found an adversarial example]
        """


        r1 = torch.rand(size=(self.particle_coordinates.shape[0], self.width*self.height*self.channelNb), device=device)
        r2 = torch.rand(size=(self.particle_coordinates.shape[0], self.width*self.height*self.channelNb), device=device)
        self.velocities = (self.particle_inertia * self.velocities + self.cognitive_weight * r1 * (self.particle_best_pos-self.particle_coordinates) + self.social_weight * r2 * (self.best_particle_position-self.particle_coordinates))
        self.particle_coordinates += (self.velocities)

        # Make sure we didn't go over the inf_norm:
        self.particle_coordinates = torch.where(self.particle_coordinates > self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.particle_coordinates)
        self.particle_coordinates = torch.where(self.particle_coordinates < self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.particle_coordinates)
        self.particle_coordinates.clamp(0,1)

        # Now we reavaluate the fitness at their new positions:
        # (we need a copy of the old fitness vals to see if we improved the particles pers. best):
        tmp = self.particle_fitness.clone()

        self.update_fitness()

        # Now let's get the best swarm fitness and position:
        newfitness , newpos= torch.max(self.particle_fitness,0)
        if newfitness > self.swarm_best_fitness:
            self.swarm_best_fitness = newfitness
            self.best_particle_position = self.particle_coordinates[newpos.item()].clone()
            self.update_predicted_label()
            if self.predicted_label != self.TRUECLASS:
                logger.debug("[Img. Nr: {}][E:{}] \t L2: {:.4f} \t predicted: {} \t should be: {} \t fitness: {:.4f}".format(self.img_id ,epoch, torch.norm(self.best_particle_position - self.target_image.view(1,-1)).item(),self.predicted_label.item(), self.TRUECLASS.item(), self.swarm_best_fitness))
                return True
            
        # And finally we update the particle's personal best values:
        # This is a boolean tensor of shape [Nbparticles] with True if the new fitness is better than the old one.
        # In this case we update the personal best location to the new particle location
        mask = (self.particle_fitness > tmp)
        self.particle_best_pos[mask] = self.particle_coordinates[mask]
        
        if epoch % 10 == 0:
            # Peridically print stats
            logger.debug("[Img. Nr: {}][E:{}] \t L2: {:.4f} \t predicted: {} \t should be: {} \t fitness: {:.4f}".format(self.img_id ,epoch, torch.norm(self.best_particle_position - self.target_image.view(1,-1)).item(),self.predicted_label.item(), self.TRUECLASS.item(), self.swarm_best_fitness))
            pass
        return False


    def eval(self):
        """Gets the raw model output for the batch of particles

        Returns:
            [torch.Tensor]: [tensor of shape [nbparticles, nbclasses], containing the confidence scores for the swarm's particles]
        """
        model_input = self.particle_coordinates[:,None,:].view(self.particle_coordinates.shape[0], self.channelNb, self.width, self.height)
        with torch.no_grad():
            output = self.model(model_input)
        return output


    def update_fitness(self, c=5):
        """Updates the fitness for all the swarm's particles

        Args:
            c (int, optional): [Regularization factor to balance confidence and L2 scores]. Defaults to 5.

        Returns:
            [Boolean]: True if we improved the swarm's best values
        """
        adv_label = self.eval()
        

        normdist = torch.norm(self.particle_coordinates - self.target_image.view(1,-1), dim=1)
        conf = (adv_label[:,self.TRUECLASS].T)[0]
        fitness = -conf - c*normdist
        self.particle_fitness = fitness


    def get_l2(self, im=None):
        if (im == None): im = self.best_particle_position
        return torch.norm(im - self.target_image.view(1,-1), dim=1).item()	

    def reduce(self):
        """Performs a simple iterative reduction algorithm to reduce the L2 norm once an adverserial example has been found
        """
        self.before_reduce = self.best_particle_position.clone()
        #Reduction of distance between original image and adversarial example.
        reshaped_target_image = self.target_image.view(self.best_particle_position.shape)
        perturbed_indices = (reshaped_target_image != self.best_particle_position) 
        previous_pos = self.best_particle_position.clone()
        factor = 0.2
        retry = 1
        MAXTRY = 3
        for i in range(100):
            if (self.predicted_label != self.TRUECLASS):
                diff = factor*(reshaped_target_image[perturbed_indices]-self.best_particle_position[perturbed_indices])
                previous_pos = self.best_particle_position.clone()
                self.best_particle_position[perturbed_indices] = self.best_particle_position[perturbed_indices] + diff
                self.update_predicted_label()
            else:
                if (i == 0):
                    logger.debug("\u001b[33m skipping reduction as attack failed \u001b[0m")
                    break
                self.best_particle_position = previous_pos
                logger.debug("[Img. Nr: {}][reduction] reduction failed at iteration {}, trying ({}/{}) again with factor {:.4f} ".format(self.img_id,i,retry, MAXTRY, factor/i))
                retry += 1
                self.update_predicted_label()
                factor /= i
                if retry == MAXTRY:
                    break