import sys
from regression_classes import *

def generate_data(n):
    X = np.random.binomial(1, expit(0), n)
    Y = np.random.binomial(1, expit(2*X), n)
    Z = np.random.binomial(1, expit(2*Y), n)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    return df

def compute_weights(df):
    # get the matrix of confounders
    Xmat = df[['X']]

    # learn the propensity score for A1
    Y = df[outcome]
    # C=np.inf makes sure that the logistic regression doesn't use
    # a penalty term
    model = LogisticRegression(C=np.inf).fit(Xmat, Y)
    A1_weights = model_A1.predict_proba(Xmat)[:,1]



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

    df = generate_data(n)

    print(df)
