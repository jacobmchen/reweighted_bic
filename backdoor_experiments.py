import sys
import pickle
from regression_classes import *
from backdoor_helper_functions import *

from skglm import GeneralizedLinearEstimator
from skglm.penalties import SCAD
from skglm.datafits import WeightedQuadratic

def run_expr(df, weights, penalized_threshold=0.01, verbose=False):
    """
    Function for simulating experiments. Need to take in a dataframe.
    For the penalized score, we fit a single regression and see which coefficients
    in the regression are sufficiently close to 0.
    For the BIC score, we perform a forward search then a backward search
    to see which variables to include in the regression.

    The parameter penalized_threshold dictates how small in absolute value a coefficient
    needs to be to be considered as 0. This method is used in Jaman, et al.
    """
    # get the sample size
    n = len(df)

    # test a normal regression as a baseline
    regression_correct = False

    linear_model = LinearRegression(weights=weights)
    # int refers to the intercept term
    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']
    linear_model.fit(Xmat, Y)
    if verbose:
        print('estimated linear regression params:\n', linear_model.params())

    # check if linear model selected the right model (ignore intercept term)
    selected_param = [False, False, False]
    for i in range(3):
        estimated_coef = linear_model.params()[i]
        if abs(estimated_coef) > penalized_threshold:
            selected_param[i] = True
    
    if selected_param == [True, False, True]:
        regression_correct = True

    # test the penalized score (SCAD)
    penalized_correct = False

    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']

    scad_model = GeneralizedLinearEstimator(datafit=WeightedQuadratic(weights.to_numpy()), penalty=SCAD(alpha=n**(-0.25), gamma=3.7))
    scad_model.fit(Xmat, Y)
    scad_params = scad_model.coef_
    if verbose:
        print('estimated scad params:\n', scad_params)

    # check if SCAD model selected the right model (ignore intercept term)
    selected_param = [False, False, False]
    for i in range(3):
        estimated_coef = scad_params[i]
        if abs(estimated_coef) > penalized_threshold:
            selected_param[i] = True
    
    if selected_param == [True, False, True]:
        penalized_correct = True

    # test the adpative LASSO penalized score
    alasso_correct = False

    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']
    # initialize the adaptive lasso model; set lambda to n^(-1/4), gamma to
    # 2.0, fit intercept to 0 because we already added a column of 1s, and
    # do not use a penalty term for the initial regression,
    # and pass in the weights needed to get rid of confounding bias
    alasso_model = AdaptiveLasso(alpha=n**(-0.25), gamma=2.0, fit_intercept=False, initial_ridge_alpha=0.0,
                                 confounding_weights=weights)
    alasso_model.fit(Xmat, Y)
    alasso_params = alasso_model.coef_
    if verbose:
        print('estimated alasso params:\n', alasso_params)

    # check if ALASSO model selected the right model (ignore intercept term)
    selected_param = [False, False, False]
    for i in range(3):
        estimated_coef = alasso_params[i]
        if abs(estimated_coef) > penalized_threshold:
            selected_param[i] = True
    
    if selected_param == [True, False, True]:
        alasso_correct = True

    bic_correct = False
    bic_choose_A2 = False
    bic_no_A1_or_A3 = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df, weights, lambda n: np.log(n), verbose=verbose)
    if verbose:
        print('selected bic model, log n penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct = True
    # verify if BIC selected A2
    if 'A2' in selected_model:
        bic_choose_A2 = True
    # verify if we failed to select A1 or A3
    if 'A1' not in selected_model or 'A3' not in selected_model:
        bic_no_A1_or_A3 = True

    bic_correct_half = False
    bic_choose_A2_half = False
    bic_no_A1_or_A3_half = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df, weights, lambda n: n**(1/2), verbose=verbose)
    if verbose:
        print('selected bic model, n^(1/2) penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct_half = True
    # verify if BIC selected A2
    if 'A2' in selected_model:
        bic_choose_A2_half = True
    # verify if we failed to select A1 or A3
    if 'A1' not in selected_model or 'A3' not in selected_model:
        bic_no_A1_or_A3_half = True

    bic_correct_three_fourths = False
    bic_choose_A2_three_fourths = False
    bic_no_A1_or_A3_three_fourths = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df, weights, lambda n: n**(3/4), verbose=verbose)
    if verbose:
        print('selected bic model, n^(3/4) penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct_three_fourths = True
    # verify if BIC selected A2
    if 'A2' in selected_model:
        bic_choose_A2_three_fourths = True
    # verify if we failed to select A1 or A3
    if 'A1' not in selected_model or 'A3' not in selected_model:
        bic_no_A1_or_A3_three_fourths = True

    if verbose:
        print()

    return (regression_correct, penalized_correct, alasso_correct, bic_correct, bic_correct_half, bic_correct_three_fourths,
            bic_choose_A2, bic_no_A1_or_A3, bic_choose_A2_half, bic_no_A1_or_A3_half, bic_choose_A2_three_fourths, bic_no_A1_or_A3_three_fourths)

if __name__ == "__main__":
    # set the seed to the input of the argument, if no input
    # seed is just 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0
    
    # set the seed
    np.random.seed(seed)

    # define the number of samples
    samples = [500, 1000, 2500, 5000, 7500, 10000]

    # keep track of how many times scad and bic
    # are correct
    linear_correct = 0
    scad_correct = 0
    alasso_correct = 0
    bic_correct = 0
    bic_correct_half = 0
    bic_correct_three_fourths = 0

    # set a flag for whether we are running the experiments with confounding
    run_with_confounding = True

    # set a flag for whether to run with verbose
    verbose = False

    # run experiments
    for sample_size in samples:
        if verbose:
            print(sample_size)

        if len(sys.argv) > 4:
            coef = float(sys.argv[3])
            confounder_variance = float(sys.argv[4])
            # generate the data with 1.5 as the coef and 0.5 as the confounder variance
            df = generate_data(sample_size, coef, confounder_variance, confounding=run_with_confounding)
        else:
            # generate the data with 1.5 as the coef and 0.5 as the confounder variance
            df = generate_data(sample_size, 1.5, 0.5, confounding=run_with_confounding)

        # check if we are clipping the weights
        if len(sys.argv) > 5 and sys.argv[5] == "clip":
            clip = '_clip'
            clip_flag = True
        else:
            clip = ''
            clip_flag = False

        # specify whether we run with confounding
        if run_with_confounding == False:
            weights = np.ones(len(df))
        else:
            # if we get an extra parameter and it is 1, use the oracle weights
            if len(sys.argv) > 2 and int(sys.argv[2]) == 1:
                weights = compute_oracle_weights(df, clip_flag)
                oracle = 'oracle_'
            else: 
                # otherwise, use the estimated weights by default
                weights = compute_weights(df, clip_flag)
                oracle = 'no_oracle_'

        if seed == 0 and sample_size == 10000:
            # save the weights to a pickle file when the sample size is 10000
            # and the seed is 0
            with open('pickle_files/backdoor_weights_' + oracle + str(coef) + '_' + str(confounder_variance) + clip + '.pkl', 'wb') as file:
                pickle.dump(weights, file)

        # run the experiments
        results = run_expr(df, weights, penalized_threshold=0.001, verbose=verbose)

        # print the results
        print(results[0])
        print(results[1])
        print(results[2])
        print(results[3])
        print(results[4])
        print(results[5])
        print(results[6])
        print(results[7])
        print(results[8])
        print(results[9])
        print(results[10])
        print(results[11])

