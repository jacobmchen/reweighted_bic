from regression_classes import *

def generate_data(n, coef, confounding=True):
    """
    Function for generating data, need to specify the sample size.
    coef specifies the strength of the dependency of Y on variables.
    """
    # define a baseline confounder
    U = np.random.normal(0, 0.5, n)
    
    # define the oracle propensity scores
    p_A1 = expit(U)
    p_A2 = expit(U)
    p_A3 = expit(U)

    # generate treatment variables
    A1 = np.random.binomial(1, p_A1, n)
    A2 = np.random.binomial(1, p_A2, n)
    A3 = np.random.binomial(1, p_A3, n)
    # add an intercept term
    intercept = np.ones(n)

    # generate the mediator variable M, which is a binary variable
    M = coef*A1 + coef*A3 + np.random.normal(0, 0.25, n)

    # generate Y based on whether we want confounding
    # Y is just a function of U and M
    if confounding == True:
        Y = coef*coef*U + coef*M + np.random.normal(0, 1, n)
    else:
        Y = coef*M + np.random.normal(0, 1, n)

    # create the dataframe, which includes the ground-truth weights that are
    # based on propensity scores
    # U is not included in the dataframe because it is unobserved
    df = pd.DataFrame({'M': M, 'A1': A1, 'A2': A2, 'A3': A3, 'int': intercept, 'Y': Y,
                       'p_A1':p_A1, 'p_A2':p_A2, 'p_A3':p_A3})

    return df

def compute_weights(df, df_p):
    """
    Compute the weights p*(A') p(M | A') / p(M | A)
    df_p is a copy of df except it contains randomized versions of the
    treatment
    """
    # get the matrix of treatments
    Xmat = df[['A1', 'A2', 'A3']]

    # get the dimension of the covariate vector
    n, d = Xmat.shape
    # add another 1 to d to account for the intercept term
    d += 1

    # learn the propensity score for M
    M = df['M']
    linear_model = LinearRegression()
    linear_model.closedform_fit(Xmat, M)

    # get the estimated parameters from the linear regression 
    theta_hat = linear_model.params()

    # get the estimates
    M_hat = np.matmul(Xmat, theta_hat)

    # compute the pdfs for each observation
    # estimate variance
    sigma_square_hat = 1/n * np.sum((M - M_hat)**2)

    # calculate the densities for denominator
    denom = 1 / np.sqrt(2 * np.pi * sigma_square_hat) * np.exp(-(M - M_hat)**2 / (2*sigma_square_hat))

    # get the matrix of prime treatments
    Xmat_p = df_p[['A1', 'A2', 'A3']]
    # get the estimates for the randomized treatments
    M_hat_p = np.matmul(Xmat_p, theta_hat)

    # compute the pdfs for each observation
    # estimate variance
    sigma_square_hat = 1/n * np.sum((M - M_hat_p)**2)

    # calculate the densities for the numerator
    numer = 1 / np.sqrt(2 * np.pi * sigma_square_hat) * np.exp(-(M - M_hat_p)**2 / (2*sigma_square_hat))

    # print('numer/denom max', np.max(numer / denom))
    # print('numer/denom min', np.min(numer / denom))
    # print('numer max', np.max(numer))
    # print('numer min', np.min(numer))
    # print('denom max', np.max(denom))
    # print('denom min', np.min(denom))

    # calculate the weights as a product of the three weights
    weights = 0.5**3 * (numer / denom)

    # print('weights min', np.min(weights))
    # print('weights max', np.max(weights))
    # print('weights sum', np.sum(weights))

    # standardize the weights
    weights_stand = weights / np.mean(weights)

    # print('weights_stand min', np.min(weights_stand))
    # print('weights_stand max', np.max(weights_stand))
    # print('weights_stand sum', np.sum(weights_stand))

    return weights_stand

def compute_oracle_weights(df, df_p, coef):
    """
    Compute the weights p*(A') p(M | A') / p(M | A) using oracle DGP of M
    df_p is a copy of df except it contains randomized versions of the
    treatment
    """
    # use the oracle DGP to get the means conditional on df values
    M_hat = coef*df['A1'] + coef*df['A3']

    # calculate the densities for denominator
    denom = 1 / np.sqrt(2 * np.pi * 0.25) * np.exp(-(df['M'] - M_hat)**2 / (2*0.25))

    # use the oracle DGP to get the means conditional on df_p values
    M_hat_p = coef*df_p['A1'] + coef*df_p['A3']

    # calculate the densities for the numerator
    numer = 1 / np.sqrt(2 * np.pi * 0.25) * np.exp(-(df['M'] - M_hat_p)**2 / (2*0.25))

    # print('numer/denom max', np.max(numer / denom))
    # print('numer/denom min', np.min(numer / denom))
    # print('numer max', np.max(numer))
    # print('numer min', np.min(numer))
    # print('denom max', np.max(denom))
    # print('denom min', np.min(denom))

    # calculate the weights as a product of the three weights
    weights = 0.5**3 * (numer / denom)

    # print('weights min', np.min(weights))
    # print('weights max', np.max(weights))
    # print('weights sum', np.sum(weights))

    # standardize the weights
    weights_stand = weights / np.mean(weights)

    # print('weights_stand min', np.min(weights_stand))
    # print('weights_stand max', np.max(weights_stand))
    # print('weights_stand sum', np.sum(weights_stand))

    return weights_stand

def bic_select_model(df, weights, bic_penalty, verbose=False):
    """
    Use the BIC score to select a model using a forward search then
    a backward search.
    """
    # define the potential coefficients to add
    possible_coefs = ['A1', 'A2', 'A3']

    # keep track of the current score and model, which is denoted
    # by a set containing the coefficients we have added to our
    # model; we start with just an empty list
    cur_score = None
    cur_model = []

    for coef in possible_coefs:
        model = LinearRegression(weights=weights, penalty=bic_penalty)
        # fit a model with the current model, the current coefficient,
        # and the intercept term
        Xmat = np.array(df[cur_model + [coef] + ['int']])
        Y = df['Y']
        model.closedform_fit(Xmat, Y)

        # get the bic score of the model
        model_score = model.compute_bic()

        # check if the score of this model is better than the current one
        if cur_score == None or cur_score > model_score:
            cur_score = model_score
            cur_model = cur_model + [coef]

    # keep track of the coefficients that we are removing
    to_remove = []
    for coef in cur_model:
        model = LinearRegression(weights=weights, penalty=bic_penalty)
        # fit a model with the current model, without the coefficient to try
        # removing, and without the coefficients we decided to remove
        Xmat = np.array(df[list(set(cur_model) - set(to_remove) - set([coef])) + ['int']])
        Y = df['Y']
        model.closedform_fit(Xmat, Y)
    
        # get the bic score of the model
        model_score = model.compute_bic()

        if cur_score > model_score:
            cur_score = model_score
            to_remove = to_remove + [coef]

    final_model = list(set(cur_model) - set(to_remove))

    return final_model


