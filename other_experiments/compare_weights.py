"""
Code defining a function that compares the reweighted BIC directly between using 
estimated and oracle weights.

Import this function to backdoor_experiments.py for use.
"""

def compare_weights():
    """
    Run simulations for comparing the BIC score when we use oracle vs.
    estimated weights.
    """
    model1_diffs = []
    model2_diffs = []
    bic_comp_ora = []
    bic_comp_est = []
    weights_rmse = []
    sample_sizes = [100, 1000, 2500, 5000, 7500, 10000, 20000]

    # see if can find DGP such that log n as penalty term
    # doesn't work, but sqrt(n) as penalty term works
    for size in sample_sizes:
        df = generate_data(size, 1.5, confounding=True)

        model1 = df[['A1', 'A2', 'int']]
        model2 = df[['A1', 'int']]

        est_weights = compute_weights(df)
        ora_weights = compute_oracle_weights(df)

        rmse = np.sqrt(np.mean((est_weights - ora_weights)**2))
        weights_rmse.append(rmse)

        A = -0.9 * np.sum(est_weights - ora_weights)
        # print('A', A)

        ora_model = LinearRegression(weights=ora_weights, penalty=lambda n: np.log(n))
        # fit a model with all terms
        Xmat = np.array(model1)
        Y = df['Y']
        ora_model.closedform_fit(Xmat, Y)

        # get the bic score of the model
        ora_model_score = ora_model.compute_bic()

        # fit a model with all terms
        Xmat = np.array(model2)
        Y = df['Y']
        ora_model.closedform_fit(Xmat, Y)

        # get the bic score of the model
        ora_model_score2 = ora_model.compute_bic()

        bic_comp_ora.append(ora_model_score - ora_model_score2)

        est_model = LinearRegression(weights=est_weights, penalty=lambda n: np.log(n))
        # fit a model with all terms
        Xmat = np.array(model1)
        Y = df['Y']
        est_model.closedform_fit(Xmat, Y)

        # get the bic score of the model
        est_model_score = est_model.compute_bic()

        # fit a model with all terms
        Xmat = np.array(model2)
        Y = df['Y']
        est_model.closedform_fit(Xmat, Y)

        # get the bic score of the model
        est_model_score2 = est_model.compute_bic()

        bic_comp_est.append(est_model_score - est_model_score2)

        model1_diffs.append(est_model_score - ora_model_score)

        model2_diffs.append(est_model_score2 - ora_model_score2)

    print('sample sizes', sample_sizes)
    print('oracle bic comp', bic_comp_ora)
    print('estimated bic comp', bic_comp_est)
    print('model 1 diffs', model1_diffs)
    print('model 2 diffs', model2_diffs)
    print('weights rmse', weights_rmse)
    print('1/sqrt(n)', 1/np.sqrt(sample_sizes))
    print('sqrt(n)', np.sqrt(sample_sizes))
 
