# Code for Experiments of the Reweighted BIC for Model Selection

This repository contains code that runs two simulations that each aim to compare the effectiveness of the reweighted BIC method against penalized regression methods (SCAD and adaptive LASSO) for model selection.

There are two data-generating processes that we simulate: the backdoor graph and the frontdoor graph.

This README file gives a high-level summary of the files in this repository and how we ran the experiments.

TLDR: 

## Files Used in Both Simulations

- ```regression_classes.py``` contains classes that implement ordinary least squares regression, linear regression with the SCAD penalty, and linear regression with the adaptive LASSO penalty. The ordinary least squares regression is implemented via both gradient descent and its closed form solution. It also computes the BIC score using various penalty functions. The other regressions are implemented via gradient descent. 
- ```process_simulation_output.py``` takes the outputs from running the simulations using SLURM and processes them into Python pickle files.
- ```plot_results.py``` takes the pickle files from above and plots them using matplotlib.

## The Backdoor Graph

- ```backdoor_helper_functions.py``` contains code for the data-generating process of the backdoor graph, computing estimated and oracle weights, and code implementing the greedy selection procedure with the reweighted BIC for model selection.
- ```backdooor_experiments.py``` contains code executing the simulations and evaluating whether each model selects the right model under different sample sizes for one repetition of each sample size.
- ```backdoor_simulations.sh``` contains shell code for executing 200 instances of the simulations in parallel when using estimated weights. The shell code works for clusters using the SLURM workload manager.
- ```backdoor_simulations_oracle.sh```contains shell code for the same as above except the weights are oracle weights.

## The Frontdoor Graph

- ```frontdoor_helper_functions.py``` is the analog to ```backdoor_helper_functions.py``` except everything is defined with respect to the frontdoor graph.
- ```frontdoor_experiments.py``` is the analog to ```backdoor_experiments.py```.
- ```frontdoor_simulations.sh``` is the analog to ```backdoor_simulations.sh```.
- ```frontdoor_simulations_oracle.sh``` is the analog to ```backdoor_simulations_oracle.sh```.

## Miscellaneous Files

- The folder ```figures``` contains the outputs of ```plot_results.py```.
- The folder ```pickle_files``` contains the outputs of ```process_simulation_output.py```.
- The folder ```other_experiments``` contains code for other experiments we ran that did not contribute to the final simulations.