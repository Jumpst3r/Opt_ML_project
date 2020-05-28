
# This file implements the PSO algorithm in a classical sequential manner

import random
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger()

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

    def update_fitness(self, c=5):
        """Updates the fitness for the current particle

        Args:
            c (int, optional): [Regularization factor to balance confidence and L2 scores]. Defaults to 5.

        Returns:
            [Boolean]: True if we improved the swarm's best values
        """
        improved_global = False
        adv_label = self.swarm.model(self.coordinates[None,:].view(1, self.swarm.channelNb, self.swarm.width, self.swarm.height))
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
    """ This class implements the swarm
    """
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
        self.target_image = target_image
        self.width = self.target_image.shape[2]
        self.height = self.target_image.shape[3]
        self.channelNb = self.target_image.shape[1]

        # We know that the target image is correctly classified, retrieve the true label
        modelout = model(self.target_image)
        self.INITCONF, self.TRUECLASS = torch.max(modelout, 1)
        self.predicted_label = self.TRUECLASS
        
        self.img_id = img_id

        inital_particle_positions = [self.target_image.view(1,self.width*self.height*self.channelNb).clone() for _ in range(nb_particles)]
        masks = [torch.randint_like(torch.Tensor(size=(1, self.width*self.height*self.channelNb)), 0, 2,device=device)==0 for _ in range(nb_particles)] # 1/10 chance of changing a pixel
        
        for p,m in zip(inital_particle_positions, masks): 
            p[m] = p[m] + 0.3 * (-2 * torch.rand(size=p[m].shape, device=device)+1)
        velocities = [torch.normal(0,1,size=(1, self.width*self.height*self.channelNb),device=device) for _ in range(nb_particles)]

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


    def step(self, epoch=0):
        """Performs one optimization step, consisting of particle position update followed by new position fitness updates.

        Args:
            epoch (int, optional): [Epoch identifier]. Defaults to 0.

        Returns:
            [Boolean]: [True if we found an adversarial example]
        """
        for p in self.particles:
            r1 = torch.rand(size=(1, self.width*self.height*self.channelNb), device=device)
            r2 = torch.rand(size=(1, self.width*self.height*self.channelNb), device=device)
            p.velocity = (self.particle_inertia * p.velocity + self.cognitive_weight * r1 * (p.best_local_arg-p.coordinates) + self.social_weight * r2 * (self.best_particle_position-p.coordinates))
            p.coordinates += (p.velocity)

            # Make sure we didn't go over the inf_norm:
            p.coordinates = torch.where(p.coordinates > self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) + self.inf_norm, p.coordinates)
            p.coordinates = torch.where(p.coordinates < self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, self.target_image.view(1,self.width *self.height*self.channelNb) - self.inf_norm, p.coordinates)
            p.coordinates.clamp(0,1)
            p.update_fitness()
            self.update_predicted_label()
        if epoch % 10 == 0:
            adv_pred = (self.model(self.best_particle_position[None,:].view(1, self.channelNb,self.width, self.height)))
            _ , idx = torch.max(adv_pred, 1)
            self.predicted_label = idx
            logger.debug("[Img. Nr: {}][E:{}] \t L2: {:.4f} \t predicted: {} \t should be: {} \t fitness: {:.4f}".format(self.img_id ,epoch, torch.norm(self.best_particle_position - self.target_image.view(1,-1)).item(),self.predicted_label.item(), self.TRUECLASS.item(), self.global_best_val.item()))
        return self.predicted_label != self.TRUECLASS

    def update_predicted_label(self):
        """Updates the predicted label corresponding to the best known particle
        """
        adv_pred = (self.model(self.best_particle_position[None,None,:].view(1,self.channelNb,self.width,self.height)))
        _ , idx = torch.max(adv_pred, 1)
        self.predicted_label = idx        
    
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
                self.best_particle_position[perturbed_indices] += diff
                logger.debug(f"In reduce, new L2: {self.get_l2()}")
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
    def get_l2(self, im=None):
        """Returns the L2 norm of the target image with the provided argument (or the swarm's best known position if arg=None)

        Args:
            im ([torch.Tensor], optional): [The tensor to compute the L2 norm with (wrt the target image)]. Defaults to None.

        Returns:
            [Float]: [The L2 norm]
        """
        if (im == None): im = self.best_particle_position
        return torch.norm(im - self.target_image.view(1,-1), dim=1).item()	
