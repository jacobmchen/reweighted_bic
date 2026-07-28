import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

class LinearRegression():
    """
    Class for a simple linear regression using gradient descent
    """
    def __init__(self, learning_rate=0.1, weights=[], penalty=lambda n: np.log(n)):
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
        self.penalty = penalty

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
        self.bic = self._compute_bic(Y_hat, Y, n, d)

    def closedform_fit(self, Xmat, Y):
        """
        Fit a linear regression using the closed form solution.
        """
        # get dimensions of the matrix
        n, d = Xmat.shape
        
        # if we are supplied weights, fit a reweighted regression
        if len(self.weights) > 0:
            W = np.diag(self.weights)
            theta = np.linalg.solve(Xmat.T @ (W @ Xmat),
                                    Xmat.T @ (W @ Y))
            self.theta = theta.copy()
        # otherwise, fit a regular regression
        else:
            theta = np.linalg.solve(Xmat.T @ Xmat, Xmat.T @ Y)
            self.theta = theta.copy()
        
        # compute the BIC score
        Y_hat = np.matmul(Xmat, self.theta)
        self.bic = self._compute_bic(Y_hat, Y, n, d)

    def _compute_bic(self, Y_hat, Y, n, d):
        error_variance = 0
        for i in range(n):
            update = (Y_hat[i] - Y[i])**2
            if len(self.weights) > 0:
                update *= self.weights[i]
            error_variance += update
        error_variance = 1/n * error_variance
        bic = n * np.log(error_variance) + d * self.penalty(n)

        return bic

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
    the penalty. The derivative of the penalty term is lambda * omega * sgn(theta)
    where lambda is a prespecified hyperparameter and omega is the reciprocal of the
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
