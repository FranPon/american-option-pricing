import numpy as np
from sklearn.linear_model import Ridge

# X is a 2D array of shape (n_stocks, paths ) 
def fit_log_standardizer(X, eps=1e-12):
    lg = np.log(X)
    mean = lg.mean(axis=1, keepdims=True)
    std = lg.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)     # at t=0 the std is 0
    return mean, std

def apply_log_standardizer(X, stats):
    mean, std = stats
    return (np.log(X) - mean) / std

def fit_regression_coefficients(X, y, ridge_alpha):
    X = np.asarray(X)
    y = np.asarray(y, dtype=X.dtype)

    if X.shape[0] == 0:
        return np.zeros( X.shape[1], dtype=float)

    if ridge_alpha == 0.0:
        return np.linalg.lstsq( X, y, rcond=None)[0]

    model = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="lsqr", copy_X=False)

    model.fit(X, y)

    return model.coef_
