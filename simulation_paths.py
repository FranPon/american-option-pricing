import numpy as np

# Simulated trajectories for multiple assets following GBM
def simulate_gbm_multi(mu, sigma, T, dt, paths, nb_stocks, sim_seed, initial_prices):
    np.random.seed(sim_seed)
    n_steps = int(round(T / dt))
    dtype = np.float32 
    mu = np.broadcast_to(np.asarray(mu, dtype=dtype), (nb_stocks,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=dtype), (nb_stocks,))
    initial_prices = np.broadcast_to(np.asarray(initial_prices, dtype=dtype), (nb_stocks,))
    # Independent Gaussian shocks stored in float32.
    Z_independent = np.empty((n_steps, paths, nb_stocks),dtype=dtype)
    for t in range(n_steps):
        Z_independent[t] = np.random.randn( paths, nb_stocks)

    dW = np.sqrt(float(dt)) * Z_independent
    log_returns = ( (mu - 0.5 * sigma**2)[None, None, :] * dt + sigma[None, None, :] * dW )
    log_price = np.concatenate([ np.zeros((1, paths, nb_stocks), dtype=dtype), np.cumsum(log_returns, axis=0, dtype=dtype)], axis=0)

    asset = initial_prices[None, None, :] * np.exp(log_price)
    asset = np.transpose(asset, (0, 2, 1))  # (n_steps + 1, nb_stocks, paths)
    return asset 
