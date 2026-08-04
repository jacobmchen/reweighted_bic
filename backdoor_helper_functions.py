from regression_classes import *

def generate_data(n, coef, confounding=True):
    """
    Function for generating data, need to specify the sample size.
    coef specifies the strength of the dependency of Y on variables.
    """
    # define a baseline confounder
    C = np.random.normal(0, 0.5, n)
    C2 = np.random.normal(1, 5, n)
    
    # define the oracle propensity scores
    p_A1 = expit(C)
    p_A2 = expit(C)
    p_A3 = expit(C)

    # generate treatment variables
    A1 = np.random.binomial(1, p_A1, n)
    A2 = np.random.binomial(1, p_A2, n)
    A3 = np.random.binomial(1, p_A3, n)
    # add an intercept term
    intercept = np.ones(n)

    # generate Y based on whether we want confounding
    # A1 and A3 are true causes while A2 is not
    if confounding == True:
        Y = coef*C + coef*A1 + 0*A2 + coef*A3 + np.random.normal(0, 1, n)
    else:
        Y = coef*A1 + 0*A2 + coef*A3 + np.random.normal(0, 1, n)

    # create the dataframe, which includes the ground-truth weights
    df = pd.DataFrame({'C': C, 'C2': C2, 'A1': A1, 'A2': A2, 'A3': A3, 'int': intercept, 'Y': Y,
                       'p_A1':p_A1, 'p_A2':p_A2, 'p_A3':p_A3})

    return df

def compute_weights(df):
    """
    Compute the weights p(A1, A2, A3 | C) = p(A1 | C)p(A2 | C)p(A3 | C)
    """
    # get the matrix of confounders
    Xmat = df[['C']]
    Xmat_wrong = df[['C2']]

    # learn the propensity score for A1
    Y = df['A1']
    # C=np.inf makes sure that the logistic regression doesn't use
    # a penalty term
    model_A1 = LogisticRegression(C=np.inf).fit(Xmat, Y)
    A1_weights = model_A1.predict_proba(Xmat)[:,1]

    Y = df['A2']
    model_A2 = LogisticRegression(C=np.inf).fit(Xmat, Y)
    A2_weights = model_A2.predict_proba(Xmat)[:,1]
    
    Y = df['A3']
    model_A3 = LogisticRegression(C=np.inf).fit(Xmat, Y)
    A3_weights = model_A3.predict_proba(Xmat)[:,1]

    # calculate the weights as a product of the three weights
    weights = (df['A1'] * A1_weights + (1-df['A1']) * (1-A1_weights)) \
                * (df['A2'] * A2_weights + (1-df['A2']) * (1-A2_weights)) \
                * (df['A3'] * A3_weights + (1-df['A3']) * (1-A3_weights))
    
    # we want the inverse weights
    weights = 0.5**3 / weights

    # standardize the weights
    weights_stand = weights / np.mean(weights)

    return weights_stand

def compute_oracle_weights(df):
    """
    Compute the weights p(A1, A2, A3 | C) = p(A1 | C)p(A2 | C)p(A3 | C) using
    the oracle probabilities of getting assigned treatment.
    """
    # calculate the weights as a product of the three weights
    weights = (df['A1'] * df['p_A1'] + (1-df['A1']) * (1-df['p_A1'])) \
                * (df['A2'] * df['p_A2'] + (1-df['A2']) * (1-df['p_A2'])) \
                * (df['A3'] * df['p_A3'] + (1-df['A3']) * (1-df['p_A3']))
    
    # we want the inverse weights
    weights = 0.5**3 / weights

    # standardize the weights
    weights_stand = weights / np.mean(weights)

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

        if verbose:
            print('compare', cur_model, 'vs.', cur_model+[coef])
            print(cur_model, 'score:', cur_score)
            print(cur_model+[coef], 'score:', model_score)
            print(cur_model+[coef], 'coefs:', model.params())
        
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

        if verbose:
            print('compare', cur_model, 'vs.', list(set(cur_model) - set(to_remove) - set([coef])))
            print(cur_model, 'score:', cur_score)
            print(list(set(cur_model) - set(to_remove) - set([coef])), 'score:', model_score)
            print(list(set(cur_model) - set(to_remove) - set([coef])), 'coefs:', model.params())

        if cur_score > model_score:
            cur_score = model_score
            to_remove = to_remove + [coef]

    final_model = list(set(cur_model) - set(to_remove))

    if verbose:
        print()

    return final_model


