# Opt_ML_project
A repository for our project for the optimization for machine learning course

## Files

The file `PSO.py` implements a particle swarm optimizer as a pytorch module. The other file simply generates a toy dataset (two gaussian blobs) and trains a simple model with the aforementioned optimizer.

Random Todo's:

1) The current implementation of the PSO algorithm is very inefficient. (Lot's of sequential iterations, copying etc). Would be cool if the particle position updates could be done all at the same time using some tensor operations
2) Test if it scales to bigger models (for example a simple CNN which could be used with the MNIST dataset)
3) Once these things are done: Compare the performance (computational cost, "convergence" speed, stochastic behavior etc) to well known first order optimizers (GD + variants)
4) Write report ?
