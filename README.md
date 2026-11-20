# Code for Experiments of the Reweighted BIC for Model Selection

This repository contains code that runs two simulations that each aim to compare the effectiveness of our proposed reweighted BIC method against penalized regression methods (SCAD and adaptive LASSO) for model selecting direct causes of an outcome.

There are two data-generating processes that we simulate: the backdoor graph and the frontdoor graph.

This README file gives a high-level summary of the files in this repository and what each of them do.

## Files Used in Both Simulations

- ```regression_classes.py``` contains classes that implement ordinary least squares regression, linear regression with the SCAD penalty, and linear regression with the adaptive LASSO penalty. The ordinary least squares regression is implemented via both gradient descent and its closed form solution. The implementation for the ordinary least squares regression also allows for computations of the BIC score using various scalings of the penalty term. The other regressions are implemented via gradient descent. 
- ```process_simulation_output.py``` takes the outputs from running the simulations and processes them into Python pickle files.
- ```plot_results.py``` takes the pickle files from above and plots them using matplotlib.
- ```plot_weights.py``` makes histograms of the frontdoor and backdoor weights based on the parameters used to run the simulations. This is for the purposes of debugging why model selection is inaccurate when weights are inaccurate.

## The Backdoor Graph

- ```backdoor_helper_functions.py``` contains code for the data-generating process of the backdoor graph, computing estimated and oracle weights, and code implementing the greedy selection procedure with the reweighted BIC for model selection.
- ```backdooor_experiments.py``` contains code executing the simulations and evaluating whether each model selects the right model under different sample sizes for one repetition of each sample size. To run this file, we can specify some parameters on the command line. The first parameter is the seed to use for the experiments. The second parameter is a 0 or 1 flag on whether to use the oracle weights. The third and fourth parameters specify the strength of association between treatment and outcome and the variance of the baseline confounder, respectively. The default values are 1.5 and 0.5, respectively.
- ```backdoor_simulations_(.).sh``` contains shell code for executing 200 instances of the simulations in parallel when using estimated weights. The shell code works for clusters using the SLURM workload manager. The (.) in the filename indicates that parameters of the simulations must be specified. We must specify whether to use oracle weights, the coefficient of the true causes, and the variance of the baseline confounder.

## The Frontdoor Graph

- ```frontdoor_helper_functions.py``` is the analog to ```backdoor_helper_functions.py``` except everything is defined with respect to the frontdoor graph.
- ```frontdoor_experiments.py``` is the analog to ```backdoor_experiments.py```. It also takes four parameters on the command line. The first parameter is the seed to use for the experiments. The second parameter is a 0 or 1 flag on whether to use the oracle weights. The third and fourth parameters specify the strength of association between treatment and outcome and the variance of the mediator, respectively. The default values are 1.5 and 2.875, respectively.
- ```frontdoor_simulations_(.).sh``` is the analog to ```backdoor_simulations_(.).sh```. Rather than specifying the variance of the baseline confounder though, the last parameter is the variance of the mediator.

## Miscellaneous Files

- The folder ```figures``` contains the outputs of ```plot_results.py```.
- The folder ```pickle_files``` contains the outputs of ```process_simulation_output.py```.
- The folder ```other_experiments``` contains code for other experiments we ran that ultimately did not contribute to the final simulations.
