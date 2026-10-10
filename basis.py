import numpy as np
from itertools import combinations_with_replacement
from math import comb
from scipy.stats import norm
from standardization_ridge import fit_log_standardizer, apply_log_standardizer

# Polynomial basis of total degree up to max_deg
def poly_basis_ram(assets, max_deg):
    nb_stocks, paths = assets.shape 
    nb_basis = comb(nb_stocks + max_deg, max_deg) # Monomials for polynomial of total degree up to max_deg in nb_stocks variables
    dtype = np.float32
    assets = np.asarray(assets, dtype=dtype)
    basis = np.empty((nb_basis, paths), dtype=dtype)  
    monomial_index = 0
    for deg in range(max_deg + 1):
        for indices in combinations_with_replacement(range(nb_stocks), deg):
            basis[monomial_index] = np.prod( assets[list(indices)], axis=0, dtype=dtype)
            monomial_index += 1
    return basis


# Indicator function basis
def indicator_basis_ram(assets, P, stats=None):
    nb_stocks, paths = assets.shape
    # log-norm CDF
    log_asset = np.log(assets)
    if stats is None:
        mu = np.mean(log_asset, axis=1, keepdims=True)
        std = np.std(log_asset, axis=1, keepdims=True)
        stats = (mu, std)
    else:
        mu, std = stats
    std_safe = np.maximum(std, 1e-8)
    phi_matrix = norm.cdf((log_asset - mu) / std_safe) # shape: (n_times, paths, nb_stocks)
    phi_matrix = np.minimum( phi_matrix, np.nextafter(1.0, 0.0)) 
    partition = np.linspace(0, 1, P + 1)
    lower_bounds = partition[:-1]
    upper_bounds = partition[1:]
    indicators = ( (phi_matrix[..., None] >= lower_bounds) & (phi_matrix[..., None] < upper_bounds) ).astype(np.int8) # (nb_stocks, paths, P)
    indicators = np.transpose(indicators, (0, 2, 1)) # (nb_stocks, P, paths)
    basis = indicators[0] # first asset partition (P, paths)

    for k in range(1, nb_stocks):
        basis = np.einsum( "iq,jq->ijq", basis, indicators[k] ) # mixed products of indicators for all assets
        basis = basis.reshape(-1 , paths)

    return basis, stats

# Herrera basis
def get_basis_herrera_random( assets, time, activation, strike_price, payoff, interest_rate, interval, A, b):
    discounted_payoff = payoff(assets, strike_price) * np.exp( -interest_rate * time * interval) 
    X = np.concatenate( [assets, discounted_payoff[None, :]], axis=0) # Additional input
    random_feature = np.einsum("hd,dq->hq", A, X) + b[:,None]
    activ = np.asarray( activation(random_feature), dtype=np.float32)
    const = np.ones((1, assets.shape[1]), dtype=np.float32)
    basis = np.concatenate([const, activ], axis=0)

    return basis

# Random basis
def get_basis_random(assets, activation, A, b, stats = None):
    # assets has shape ( nb_stocks, paths)
    if stats is None:
        stats = fit_log_standardizer(assets)
    # Standardization
    X_standardized = apply_log_standardizer(assets, stats)
    
    random_features = np.einsum("hd,dq->hq", A, X_standardized) + b[:, None] # (nneurons - 1, paths )
    activ = np.asarray( activation(random_features), dtype=np.float32)
    const = np.ones( (1, assets.shape[1]), dtype=np.float32)
    basis = np.concatenate([const, activ], axis=0) # shape: (nneurons, paths)

    return basis, stats