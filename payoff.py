import numpy as np

# Assets is a 2D array of shape (nb_stocks, paths) and payoff is a 1D array of shape (paths)
def max_call_ram(assets, strike):
    return np.maximum(np.max(assets, axis=0) - strike, 0.0)

def min_put_ram(assets, strike):
    return np.maximum(strike - np.min(assets, axis=0), 0.0)

def basket_call_ram(assets, strike):
    return np.maximum(np.mean(assets, axis=0) - strike, 0.0)

def basket_put_ram(assets, strike):
    return np.maximum(strike - np.mean(assets, axis=0), 0.0)

def geometric_put_ram(assets, strike) :
    geometric_mean = np.exp(np.mean(np.log(assets), axis=0))
    return np.maximum(strike - geometric_mean, 0.0)
