import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression

class LinearRegression():
    """
    Class for a simple linear regression using gradient descent
    """

    def __init__(self, learning_rate=0.1, weights=[]):
        """
        Constructor for the class. The learning rate specifies how
        quickly we move at each iteration of updating the derivative.
        The weights are for a reweighted regression. If the weights
        are None, then fit a regular regression.
        """
        self.learning_rate = learning_rate
        self.theta = None
        self.bic = 0
        self.weights = weights

    def _calculate_gradient(self, Xmat, Y, theta_t):
        """
        Private function for computing the gradient.
        Xmat is a numpy matrix and Y is a numpy array.
        theta_t represents the current values of the coefficients in the
        regression and is also a numpy array.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape

        # set up the gradient for each coefficient
        grad_vec = np.zeros(d)

        # calculate the predictions Y_hat at the current values of theta
        Y_hat = np.matmul(Xmat, theta_t)

        # we are using the mean squared error as the loss function, therefore
        # dL/d(theta_i) = - 2/n \sum_i=1^n (Y_i - Y_hat_i)X_ij

        # for each coefficient that we are fitting
        for j in range(d):
            # compute the gradient for that vector
            gradient = 0
            # for each row of data
            for i in range(n):
                # calculate the partial derivative for coefficient j
                # at row i
                gradient_update = (Y[i] - Y_hat[i])*Xmat[i][j]
                if len(self.weights) > 0:
                    # if we have weights, then multiply them onto the update
                    gradient_update = gradient_update * self.weights[i]
                # update the gradient
                gradient = gradient + gradient_update
            gradient = gradient * -2 / n
            grad_vec[j] = gradient

        return grad_vec

    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Fit a linear regression using gradient descent. The tolerance specifies
        the difference at which we will terminate the program since the theta's
        are not changing anymore.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape

        # self.theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xmat.T, Xmat)), Xmat.T), Y)
        #
        # initialize guesses for the thetas randomly
        theta = np.random.uniform(-5, 5, d)
        theta_new = np.random.uniform(-5, 5, d)

        # initialize an iteration counter
        iteration = 0

        while iteration < max_iterations:
            # calculate the gradients at the current theta
            grad_vec = self._calculate_gradient(Xmat, Y, theta)

            # update the thetas that we have learned using the gradients
            theta_new = theta - self.learning_rate*grad_vec

            # check if we have achieved our tolerance
            tolerance_achieved = True
            difference = theta_new - theta
            for i in range(d):
                if abs(difference[i]) > tolerance:
                    tolerance_achieved = False
                    # there is at least one theta that is still far away
                    break

            # we have achieved our tolerance, so exit gradient descent
            if tolerance_achieved:
                break

            # update the old theta
            theta = theta_new.copy()

            # update the iteration count
            iteration += 1

        # update the attribute theta
        self.theta = theta.copy()

        # compute the BIC score
        Y_hat = np.matmul(Xmat, self.theta)
        error_variance = 0
        for i in range(n):
            update = (Y_hat[i] - Y[i])**2
            if len(self.weights) > 0:
                update *= self.weights[i]
            error_variance += update
        error_variance = 1/n * error_variance
        self.bic = n * np.log(error_variance) + d * np.log(n)

    def params(self):
        return self.theta

    def compute_bic(self):
        # return the BIC score, which was computed at fit
        return self.bic

class LinearRegressionSCAD(LinearRegression):
    """
    Class for linear regression with the SCAD penalty.

    The choice for the parameter a=3.7 is numerically shown to be good.
    To get oracle properties, lambda must satisfy lambda_n -> 0 and n^(1/2)*lambda_n -> infty
    """
    def __init__(self, learning_rate=0.01, weights=[], lambdaa=0.1, a=3.7):
        super().__init__(learning_rate, weights)
        self.lambdaa=lambdaa
        self.a=a

    def _calculate_gradient(self, Xmat, Y, theta_t):
        """
        Private function for computing the gradient with the SCAD penalty.
        Xmat is a numpy matrix and Y is a numpy array.
        theta_t represents the current values of the coefficients in the
        regression and is also a numpy array.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape

        # set up the gradient for each coefficient
        grad_vec = np.zeros(d)

        # calculate the predictions Y_hat at the current values of theta
        Y_hat = np.matmul(Xmat, theta_t)

        # we are using the mean squared error as the loss function, therefore
        # dL/d(theta_i) = - 2/n \sum_i=1^n (Y_i - Y_hat_i)X_ij

        # for each coefficient that we are fitting
        for j in range(d):
            # compute the gradient for that vector
            gradient = 0
            # for each row of data
            for i in range(n):
                gradient_update = (Y[i] - Y_hat[i])*Xmat[i][j]
                if len(self.weights) > 0:
                    # if we have weights, then use them
                    gradient_update = gradient_update * self.weights[i]
                gradient = gradient + gradient_update
            gradient = gradient * -2/n

            # add the SCAD penalty
            sgn_theta_t = 1 if theta_t[j] > 0 else -1
            if abs(theta_t[j]) <= self.lambdaa:
                gradient += self.lambdaa * sgn_theta_t
            elif abs(theta_t[j]) > self.lambdaa and abs(theta_t[j]) <= (self.a*self.lambdaa):
                gradient += (self.a*self.lambdaa - abs(theta_t[j]))/(self.a-1) * sgn_theta_t

            grad_vec[j] = gradient

        return grad_vec

class LinearRegressionALASSO(LinearRegression):
    """
    Class for the adaptive LASSO linear regression.

    Adaptive LASSO is a two-stage procedure where the first stage is to fit
    a regular linear regression and the second stage is to fit a regression with
    the penalty. The derivative of the penalty term is \lambda * \omega * sgn(theta)
    where \lambda is a prespecified hyperparameter and \omega is the reciprocal of the
    absolute value of the coefficient learned in the regular linear regression.

    There is technically another parameter here gamma that dictates the value of omega
    as follows: omega = 1 / abs(param)^gamma. Here, we just use gamma=1.

    To get oracle properties, lambda_n/n^(1/2) -> 0 and lambda_n*n^((gamma-1)/2) -> infty
    """
    def __init__(self, Xmat, Y, learning_rate=0.01, weights=[], lambdaa=0.1):
        super().__init__(learning_rate, weights)
        self.lambdaa=lambdaa
        # use Xmat and Y to fit a regular linear regression and set the weights
        # omega
        ols_model = LinearRegression(weights=self.weights)
        ols_model.fit(Xmat, Y)
        ols_params = ols_model.params()
        self.omega = []
        for param in ols_params:
            self.omega.append( 1 / abs(param) )

    def _calculate_gradient(self, Xmat, Y, theta_t):
        """
        Private function for computing the gradient with the ALASSO penalty.
        Xmat is a numpy matrix and Y is a numpy array.
        theta_t represents the current values of the coefficients in the
        regression and is also a numpy array.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape

        # set up the gradient for each coefficient
        grad_vec = np.zeros(d)

        # calculate the predictions Y_hat at the current values of theta
        Y_hat = np.matmul(Xmat, theta_t)

        # we are using the mean squared error as the loss function, therefore
        # dL/d(theta_i) = - 2/n \sum_i=1^n (Y_i - Y_hat_i)X_ij

        # for each coefficient that we are fitting
        for j in range(d):
            # compute the gradient for that vector
            gradient = 0
            # for each row of data
            for i in range(n):
                gradient_update = (Y[i] - Y_hat[i])*Xmat[i][j]
                if len(self.weights) > 0:
                    # if we have weights, then use them
                    gradient_update = gradient_update * self.weights[i]
                gradient = gradient + gradient_update
            gradient = gradient * -2/n

            # add the adaptive LASSO penalty
            sgn_theta_t = 1 if theta_t[j] > 0 else -1
            gradient += self.lambdaa * self.omega[j] * sgn_theta_t
            grad_vec[j] = gradient

        return grad_vec

def generate_data(n, coef, confounding=True):
    """
    Function for generating data, need to specify the sample size.
    coef specifies the strength of the dependency of Y on variables.
    """
    # define a baseline confounder
    C = np.random.normal(0, 1, n)
    
    A1 = np.random.binomial(1, expit(C), n)
    A2 = np.random.binomial(1, expit(C), n)
    A3 = np.random.binomial(1, expit(C), n)
    intercept = np.ones(n)
    if confounding == True:
        Y = coef*C + coef*A1 + 0*A2 + coef*A3 + np.random.normal(0, 1, n)
    else:
        Y = coef*A1 + 0*A2 + coef*A3 + np.random.normal(0, 1, n)

    df = pd.DataFrame({'C': C, 'A1': A1, 'A2': A2, 'A3': A3, 'int':intercept, 'Y': Y})

    return df

def compute_weights(df):
    """
    Compute the weights p(A1, A2, A3 | C) = p(A1 | C)p(A2 | C)p(A3 | C)
    """
    Xmat = np.array([df['C']]).T

    Y = df['A1']
    model_A1 = LogisticRegression(penalty=None).fit(Xmat, Y)
    A1_weights = model_A1.predict_proba(Xmat)[:,1]

    Y = df['A2']
    model_A2 = LogisticRegression(penalty=None).fit(Xmat, Y)
    A2_weights = model_A2.predict_proba(Xmat)[:,1]
    
    Y = df['A3']
    model_A3 = LogisticRegression(penalty=None).fit(Xmat, Y)
    A3_weights = model_A3.predict_proba(Xmat)[:,1]

    # calculate the weights as a product of the three weights
    weights = (df['A1'] * A1_weights + (1-df['A1']) * (1-A1_weights)) \
                * (df['A2'] * A2_weights + (1-df['A2']) * (1-A2_weights)) \
                * (df['A3'] * A3_weights + (1-df['A3']) * (1-A3_weights))
    
    # we want the inverse weights
    weights = 1 / weights

    # standardize the weights
    weights_stand = weights / np.mean(weights)

    return weights_stand

def bic_select_model(df, weights):
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
        model = LinearRegression(weights=weights)
        # fit a model with the current model, the current coefficient,
        # and the intercept term
        Xmat = np.array(df[cur_model + [coef] + ['int']])
        Y = df['Y']
        model.fit(Xmat, Y)

        # get the bic score of the model
        model_score = model.compute_bic()
        
        # check if the score of this model is better than the current one
        if cur_score == None or cur_score > model_score:
            cur_score = model_score
            cur_model = cur_model + [coef]

    # keep track of the coefficients that we are removing
    to_remove = []
    for coef in cur_model:
        model = LinearRegression(weights=weights)
        # fit a model with the current model, without the coefficient to try
        # removing, and without the coefficients we decided to remove
        Xmat = np.array(df[list(set(cur_model) - set(to_remove) - set([coef])) + ['int']])
        Y = df['Y']
        model.fit(Xmat, Y)
    
        # get the bic score of the model
        model_score = model.compute_bic()

        if cur_score > model_score:
            cur_score = model_score
            to_remove = to_remove + [coef]

    final_model = list(set(cur_model) - set(to_remove))

    return final_model

def run_expr(df, weights, penalized_threshold=0.01, verbose=False):
    """
    Function for simulating experiments. Need to take in a dataframe.
    For the penalized score, we fit a single regression and see which coefficients
    in the regression are sufficiently close to 0.
    For the BIC score, we perform a forward search then a backward search
    to see which variables to include in the regression.

    The penalized_threshold dictates how small in absolute value a coefficient
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
    # to-do: fit the model using a reweighted score instead
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

    scad_model = LinearRegressionSCAD(weights=weights, lambdaa=n**(-0.25))
    Xmat = np.array(df[['A1', 'A2', 'A3', 'int']])
    Y = df['Y']
    # to-do: fit the model using a reweighted score instead
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
    alasso_model = LinearRegressionALASSO(Xmat, Y, weights=weights, lambdaa=n**(-0.25))
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
    selected_model = bic_select_model(df, weights)
    if verbose:
        print('selected bic model:\n', selected_model)
    # verify if BIC selected the right model
    if 'A1' in selected_model and 'A3' in selected_model and 'A2' not in selected_model:
        bic_correct = True

    print()

    return (regression_correct, penalized_correct, alasso_correct, bic_correct)

if __name__ == "__main__":
    # set the seed
    np.random.seed(0)

    # set number of iterations for experiments
    num_experiments = 100
    # define the number of samples
    n = 500

    # keep track of how many times scad and bic
    # are correct
    linear_correct = 0
    scad_correct = 0
    alasso_correct = 0
    bic_correct = 0

    # set a flag for whether we are running the experiments with confounding
    run_with_confounding = True

    # run experiments
    for i in range(num_experiments):
        print('experiment number:', i)

        df = generate_data(n, 1.5, confounding=run_with_confounding)

        if run_with_confounding == False:
            weights = np.ones(len(df))
        else:
            weights = compute_weights(df)

        results = run_expr(df, weights, penalized_threshold=0.001, verbose=True)

        if results[0] == True:
            linear_correct += 1

        if results[1] == True:
            scad_correct += 1

        if results[2] == True:
            alasso_correct += 1

        if results[3] == True:
            bic_correct += 1

    # print out the experimental results
    print('linear percentage:', linear_correct/num_experiments)
    print('scad percentage:', scad_correct/num_experiments)
    print('alasso percentage:', alasso_correct/num_experiments)
    print('bic percentage:', bic_correct/num_experiments)

