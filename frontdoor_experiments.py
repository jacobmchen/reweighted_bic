from regression_classes import *
from frontdoor_helper_functions import *

def run_expr(df, df_p, weights, penalized_threshold=0.01, verbose=False):
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

    linear_model = LinearRegression()
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

    scad_model = LinearRegressionSCAD(lambdaa=n**(-0.25))
    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']
    scad_model.fit(Xmat, Y)
    if verbose:
        print('estimated scad params:\n', scad_model.params())

    # check if SCAD model selected the right model (ignore intercept term)
    selected_param = [False, False, False]
    for i in range(3):
        estimated_coef = scad_model.params()[i]
        if abs(estimated_coef) > penalized_threshold:
            selected_param[i] = True
    
    if selected_param == [True, False, True]:
        penalized_correct = True

    # test the adpative LASSO penalized score
    alasso_correct = False

    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']
    alasso_model = LinearRegressionALASSO(Xmat, Y, lambdaa=n**(-0.25))
    alasso_model.fit(Xmat, Y)
    if verbose:
        print('estimated alasso params:\n', alasso_model.params())

    # check if ALASSO model selected the right model (ignore intercept term)
    selected_param = [False, False, False]
    for i in range(3):
        estimated_coef = alasso_model.params()[i]
        if abs(estimated_coef) > penalized_threshold:
            selected_param[i] = True
    
    if selected_param == [True, False, True]:
        alasso_correct = True

    bic_correct = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df_p, weights, lambda n: np.log(n))
    if verbose:
        print('selected bic model, log n penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct = True

    bic_correct_half = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df_p, weights, lambda n: n**(1/2))
    if verbose:
        print('selected bic model, n^(1/2) penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct_half = True

    bic_correct_three_fourths = False
    # use the BIC score method to select a model
    selected_model = bic_select_model(df_p, weights, lambda n: n**(3/4))
    if verbose:
        print('selected bic model, n^(3/4) penalty:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct_three_fourths = True

    if verbose:
        print()

    return (regression_correct, penalized_correct, alasso_correct, bic_correct, bic_correct_half, bic_correct_three_fourths)

if __name__ == "__main__":
    # set the seed to the input of the argument, if no input
    # seed is just 0
    if len(sys.argv) > 1:
        seed = sys.argv[1]
    else:
        seed = 0
    
    # set the seed
    np.random.seed(seed)

    # define the number of samples
    samples = [500, 1000, 2500, 5000, 10000, 50000]

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

    # run experiments
    for sample_size in samples:
        # generate the data
        df = generate_data(sample_size, 3.5, confounding=run_with_confounding)

        # get number of rows in df
        n = len(df)

        # get a copy of the dataframe
        df_p = df.copy()

        if run_with_confounding == False:
            weights = np.ones(len(df))
        else:
            # generate prime values of the three treatments
            A1_p = np.random.binomial(1, 0.5, n)
            A2_p = np.random.binomial(1, 0.5, n)
            A3_p = np.random.binomial(1, 0.5, n)

            # replace the treatments in df with randomized versions
            df_p['A1'] = A1_p
            df_p['A2'] = A2_p
            df_p['A3'] = A3_p

            weights = compute_weights(df, df_p)
            oracle_weights = compute_oracle_weights(df, df_p)
            # print('weights rmse', np.sqrt(np.mean((weights - oracle_weights)**2)))

        # run the experiments
        results = run_expr(df, df_p, weights, penalized_threshold=0.001, verbose=False)

        # print the results
        print(results[0])
        print(results[1])
        print(results[2])
        print(results[3])
        print(results[4])
        print(results[5])

        if results[0] == True:
            linear_correct += 1

        if results[1] == True:
            scad_correct += 1

        if results[2] == True:
            alasso_correct += 1

        if results[3] == True:
            bic_correct += 1

        if results[4] == True:
            bic_correct_half += 1

        if results[5] == True:
            bic_correct_three_fourths += 1

    # print out the experimental results
    # print('linear percentage:', linear_correct/num_experiments)
    # print('scad percentage:', scad_correct/num_experiments)
    # print('alasso percentage:', alasso_correct/num_experiments)
    # print('bic percentage, log n penalty:', bic_correct/num_experiments)
    # print('bic percentage, n^(1/2) penalty:', bic_correct_half/num_experiments)
    # print('bic percentage, n^(3/4) penalty:', bic_correct_three_fourths/num_experiments)

