import sys
from regression_classes import *
from backdoor_helper_functions import *
from backdoor_experiments import *

"""
Try to create a DGP such that A2 is not significant in E[Y | A1, A2]
but A2 is significant in E[Y | A1, A2, C] while Y is a member of
the exponential family.
"""

if __name__ == "__main__":
    # set the seed to the input of the argument, if no input
    # seed is just 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0
    
    # set the seed
    np.random.seed(seed)

    n = 5000

    C = np.random.normal(0, 1, n)

    A1 = np.random.binomial(1, 0.5, n)
    A2 = np.random.binomial(1, 0.5, n)

    Y = A1 + A2*C + np.random.normal(0, 1, n)

    df = pd.DataFrame({'A1': A1, 'A2': A2, 'A2*C': A2*C, 'C': C, 'Y': Y})

    model = LinearRegression()

    Xmat = np.array(df[['A1', 'A2']])
    Y = df['Y']
    model.closedform_fit(Xmat, Y)
    # get the bic score of the model
    model_score = model.compute_bic()
    print(model.params())
    print(model_score)

    Xmat = np.array(df[['A1']])
    Y = df['Y']
    model.closedform_fit(Xmat, Y)
    # get the bic score of the model
    model_score = model.compute_bic()
    print(model.params())
    print(model_score)

