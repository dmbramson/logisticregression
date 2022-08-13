import numpy as np
import numpy.typing as npt
import pandas as pd

# Test data is iris-setosa on petal.width

EPS = np.finfo(float).eps
ALPHA = 1e-2


def logloss(
    y: npt.NDArray[np.number], yhat: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    """Compute the logloss function.

    Behavior not defined when y not in {0, 1}."""
    return -y * np.log(yhat) - (1 - y) * np.log(1 - yhat)


def logistic(X: npt.NDArray[np.number], b: npt.NDArray) -> npt.NDArray[np.number]:
    """Compute the logistic function.

    Inputs
    ----------------------------------
    x : n-by-k ndarray
    b0 : float
    b : k-by-1 ndarray
    """
    return 1 / (1 + np.exp(-(np.sum(X * b, axis=1))))


def calc_logloss_grad(
    X: npt.NDArray[np.number],
    y: npt.NDArray[np.number],
    b: npt.NDArray,
    eps: np.number = EPS,
) -> npt.NDArray[np.number]:
    """Calculate the gradient of the logloss function."""
    h = np.array([h0 if h0 > 0 else np.sqrt(eps) / 10 for h0 in b * np.sqrt(eps)])
    ds = np.zeros(h.shape)
    for i in range(0, h.shape[0]):
        zero = np.zeros(h.shape)
        zero[i] = 1
        bup = np.sum(logloss(y, logistic(X, b + (h * zero))))
        bdown = np.sum(logloss(y, logistic(X, b - (h * zero))))
        ds[i] = (bup - bdown) / (2 * np.sum(h * zero))
    return ds


def grad_descent(
    X: npt.NDArray[np.number],
    y: npt.NDArray[np.number],
    b: npt.NDArray,
    eps: np.number = EPS,
    threshold: float = 1e-2,
    alpha: float = ALPHA,
    max_iter: int = 10000,
) -> npt.NDArray[np.number]:
    """Optimize logistic regression parameters using gradient descent."""
    i = 1
    while np.linalg.norm(b) > threshold:
        grad = calc_logloss_grad(X, y, b, eps)
        step = grad * alpha
        b = b - step
        print("Iteration {}: b = {}, grad.norm = {}".format(i, b, np.linalg.norm(grad)))
        i = i + 1
        if i > max_iter:
            break
    return b


raw_data = pd.read_csv("testdata/iris.csv")

test_data = raw_data[["petal.length", "petal.width", "class"]].copy()
test_data["intercept"] = 1
cols = list(test_data.columns.values)
cols = cols[-1:] + cols[:-1]
test_data = test_data[cols]

test_data.loc[:, "class"] = test_data["class"].apply(
    lambda x: 1 if x == "Iris-setosa" else 0
)
X = test_data.drop("class", axis=1)
y = test_data["class"]

b = np.array([1, 1, 1])
yhat = logistic(X, b)
z = logloss(y, yhat)
grad = calc_logloss_grad(X, y, b)
step = grad * ALPHA
b = b - step

# Fails to converge in 10k, final gradient norm is 0.0275
# Still finds good coefficients that actually result in 100% accuracy
b = np.array([1, 1, 1])
i = 1
while np.linalg.norm(b) > 1e-2:
    grad = calc_logloss_grad(X, y, b)
    step = grad * ALPHA
    b = b - step
    print("Iteration {}: b = {}, grad.norm = {}".format(i, b, np.linalg.norm(grad)))
    i = i + 1
    if i > 10000:
        break

yhat = logistic(X, b)
z = logloss(y, yhat)  # 0.083
ypred = np.round(yhat, 0)
