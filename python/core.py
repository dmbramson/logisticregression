import numpy as np
import numpy.typing as npt
import pandas as pd

# Test data is iris-setosa on petal.width

EPS = np.finfo(float).eps

def logloss(y: npt.NDArray[np.number], yhat: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Compute the logloss function.

    Behavior not defined when y not in {0, 1}."""
    return -y*np.log(yhat) - (1-y)*np.log(1-yhat)


def logistic(X: npt.NDArray[np.number], b: npt.NDArray) -> npt.NDArray[np.number]:
    """Compute the logistic function.

    Inputs
    ----------------------------------
    x : n-by-k ndarray
    b0 : float
    b : k-by-1 ndarray
    """
    return 1/(1+np.exp(-(np.sum(X*b, axis=1))))


def calc_logloss_step(X: npt.NDArray[np.number], y: npt.NDArray[np.number], b: npt.NDArray, ) -> npt.NDArray[np.number]:
    """Calculate the gradient of the logloss function."""
    h = np.array([h0 if h0 > 0 else np.sqrt(EPS)/10 for h0 in b*np.sqrt(EPS)])
    ds = np.zeros(h.shape)
    for i in range(0, h.shape[0]):
        zero = np.zeros(h.shape)
        zero[i] = 1
        bup = np.sum(logloss(y, logistic(X, b + (h*zero))))
        bdown = np.sum(logloss(y, logistic(X, b - (h*zero))))
        ds[i] = -(bup-bdown)/(2)



raw_data = pd.read_csv('testdata/iris.csv')

test_data = raw_data[['petal.length', 'petal.width', 'class']].copy()
test_data['intercept'] = 1
cols = list(test_data.columns.values)
cols = cols[-1:] + cols[:-1]
test_data = test_data[cols]

test_data.loc[:, 'class'] = test_data['class'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)
X = test_data.drop('class', axis = 1)
y = test_data['class']

b = np.array([0, 1, 2])
yhat = logistic(X, b)
z = logloss(y, yhat)