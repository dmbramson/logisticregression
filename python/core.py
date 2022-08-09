import numpy as np

# Test data is iris-setosa on petal.width


def logloss(y: float, yhat: float) -> float:
    """Computes the logloss function.

    Behavior not defined when y not in {0, 1}."""
    return -np.log(yhat) if y == 1 else -np.log(1-yhat)
