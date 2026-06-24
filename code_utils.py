import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Utility function 

# Simulation of the underlying asset price using Geometric Brownian Motion (GBM) using antithetic variates 
def simulate_gbm( mu, sigma, T, dt, b, seed):
    np.random.seed(seed)
    N = int(T / dt)  # Number of time steps
    Z = np.random.standard_normal((b // 2, N)) # Increments
    Z = np.vstack([Z, -Z])
    BM = np.zeros((N + 1, b))
    BM[1:]= np.cumsum(Z * np.sqrt(dt), axis=1).T 
    time = np.arange(0, N + 1) * dt
    drift = (mu - 0.5 * sigma**2) * time
    S = np.exp(drift[:,np.newaxis] + sigma * BM)
    return S

# Simualted GBM trajectories for the underlying asset price : (N_bar*N+1, paths)
def simulate_gbm_nv(r, sigma, T, dt, paths, seed):
    np.random.seed(seed)
    n_steps = int(round(T / dt))
    Z = np.random.randn(n_steps, paths)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_price = np.vstack([np.zeros((1, paths)), np.cumsum(log_returns, axis=0)])
    return np.exp(log_price) 

def transformation(X):
    trans = np.log(X)
    std = np.std(X, axis=1, keepdims=True)
    std[0] = 1.0
    mean = np.mean(trans, axis=1, keepdims=True)
    return (trans - mean) / std


def ls_train(K, S0, N, train_set, discount_factors, basis_train, npaths_training, max_deg):
    # Precomputing the intrinsic value at each time step discounted to time 0 for the training set
    intrinsic_value_train = np.maximum(0, K - S0*train_set) * discount_factors[:,np.newaxis] 

    # Training phase : Backward induction compute the coefficients
    value = np.empty((N, npaths_training)) # Option value at time t for each path, initialized to 0 at maturity
    value[-1,:] = intrinsic_value_train[-1,:] # Intrinsic value at time 0, discounted to time 0
    
    train_deg = np.zeros(N-1, dtype=int) # Actual degree used for regression at each time step
    coefficients = np.zeros((N-1, max_deg + 1)) 
    continuation_value = np.zeros(( N-1, npaths_training)) 

    itm_train = intrinsic_value_train > 0 # Indices of ITM paths at each time step, discounted to time 0

    for t in range(N - 1, 0, -1):
        itm = itm_train[t,:] # Indices of ITM paths at time t, discounted to time 0
        itm_paths = np.sum(itm) # Number of in the money paths at time t

        # Setting itm True for all paths to see the effect of regression on the continuation value estimation
        #itm = np.ones_like(payoff_t, dtype=bool) # All paths are considered in the regression, even the OTM ones
        
        # Regression to estimate continuation value, hence to compute the coefficient 
        train_deg[t-1] = min(max_deg, int(itm_paths / 100) ) # Degree of the polynomial basis for regression
        value[t-1, :] = value[t, :] # Initializing the option value at time t-1 to the value at time t (continuation value without regression)
        
        if (train_deg[t-1] > 0) & (itm_paths > train_deg[t-1]):
            Y_itm = value[t, itm] 
            vander_full = basis_train[t-1, :, :train_deg[t-1]+1]
            X_itm_vander = basis_train[t-1, itm , :train_deg[t-1]+1]
            # Regression
            coefficients[t-1, :train_deg[t-1]+1], _, _, _ = np.linalg.lstsq(X_itm_vander, Y_itm, rcond=None) 
            continuation_value[t-1,:] = vander_full @ coefficients[t-1, :train_deg[t-1]+1]
            # Determine the option value at time t
            exercise = (intrinsic_value_train[t,:] > continuation_value[t-1,:]) & (itm) # Exercise condition at time t
            # The option is early-exercised if the payoff is positive and greater than the continuation value!
            value[t-1, : ] = np.where(exercise, intrinsic_value_train[t,:], value[t,:]) 
            # If the option is exercised, the value is the payoff, otherwise it's the continuation value
            # What we actually do in practice is to use the value at time t so that the approximation of the continuation
            # value do not affect the value of the "subsequence" continuation value.

    return coefficients, continuation_value, value, train_deg

# Observe that basis train and basis pricing are realizing using the train and test log
def lower_bound_ls(K, S0, T, dt, train_set, test_set, discount_factors, npaths_training, npaths_pricing, max_deg, basis_train, basis_pricing):
    N = int(T / dt)
    
    coefficients, continuation_value, value, train_deg = ls_train(K, S0, N, train_set, discount_factors, basis_train, npaths_training, max_deg)

    # Pricing phase
    continuation_value_pricing = np.zeros((N-1,npaths_pricing)) # Storing only for plotting
    exercise_payoff = np.zeros(npaths_pricing) # Store the exercise time for each path, initialized to N (never early-exercised)
    has_exercised = np.zeros(npaths_pricing, dtype=bool) # Tracking if the option has been exercised for each path

    # Computing the price
    intrinsic_value_price = np.maximum(0, K - S0*test_set) * discount_factors[:, np.newaxis] # Intrinsic value at maturity discounted to time 0
    itm_pricing = intrinsic_value_price > 0 
    
    for t in range(0, N - 1 ):
        itm = itm_pricing[t+1,:]
        itm_pricing_paths = np.sum(itm) # Number of ITM paths 
        if (train_deg[t] > 0) & (itm_pricing_paths > train_deg[t]):
            # Computing the continuation value 
            vander_pricing = basis_pricing[t, : , :train_deg[t]+1]
            continuation_value_pricing[t,:] = vander_pricing @ coefficients[t, :train_deg[t]+1]

            exercise_condition = (intrinsic_value_price[t+1,:] > continuation_value_pricing[t,:]) & itm & (has_exercised == False)
            exercise_payoff[exercise_condition] = intrinsic_value_price[t+1,exercise_condition]
            has_exercised[exercise_condition] = True 
    
    # If the option has not been exercised at any time step, the payoff is the one at maturity  
    exercise_payoff[~has_exercised] = intrinsic_value_price[-1,~has_exercised]
    # Since we are generating the paths using a variance reduction technique (antithetic variates), we need to take
    # this into account: we need to do the average of the "antithetic values", hence this is wrong option_price = np.mean(exercise_payoff)
    antithetic_sum = (exercise_payoff[:npaths_pricing//2] + exercise_payoff[npaths_pricing//2:]) / 2
    option_price = np.mean(antithetic_sum)
    #option_price = np.mean(exercise_payoff)
    immediate_exercise = max(0, K - S0) # Immediate exercise value at time 0
    option_price = max(option_price, immediate_exercise)

    uncertainty = np.std(antithetic_sum, ddof=1) / np.sqrt(npaths_pricing//2)
    # Uncertainty without variance reduction 
    #uncertainty_nv = np.std(exercise_payoff, ddof=1) / np.sqrt(npaths_pricing) 
    return option_price, uncertainty, coefficients, continuation_value, value, train_deg

# Computing the indicator function basis for time t_{i,j}
def get_indicator_basis_at_time_mod(asset_at_t, P):
    # log-norm CDF
    log_asset = np.log(asset_at_t)
    sigma_local = np.std(log_asset)
    mu_local = np.mean(log_asset)

    sigma_local_safe = 1e-8 if sigma_local == 0 else sigma_local
    phi_matrix = norm.cdf((log_asset - mu_local) / sigma_local_safe)
    
    partition    = np.linspace(0, 1, P + 1)
    lower_bounds = partition[:-1, np.newaxis]
    upper_bounds = partition[1:,  np.newaxis]
    
    basis = ((phi_matrix >= lower_bounds) & (phi_matrix < upper_bounds)).astype(np.int8)
    return basis

# Computing martingale for control variate/upper bound using Alfonsi method knowing the coefficients
def alfonsi_martingale(N, N_bar, P, discounted_asset, coefficients, paths, asset):
    martingale_increments = np.zeros((N, paths))
    for i in range(N-1, -1, -1):
        martingale_increment_i = np.zeros(paths)
        for j in range(N_bar):
            k = i * N_bar + j
            delta_A = discounted_asset[k+1, :] - discounted_asset[k, :]  
            indicators = get_indicator_basis_at_time_mod(asset[k, :], P)         
            delta_X_all = indicators * delta_A[np.newaxis, :]                    
            alpha_all = coefficients[i, j, :]                             
            martingale_increment_i += alpha_all @ delta_X_all                 
        martingale_increments[i, :] = martingale_increment_i
    martingale = np.cumsum(martingale_increments, axis=0)
    return martingale

# Training phase Alfonsi
def computing_coefficients_martingale_alfonsi(N, N_bar, P, discounted_asset, discounted_payoff, paths, asset):

    coefficients = np.zeros((N, N_bar, P))
    martingale_increments = np.zeros((N, paths))
    martingale = np.zeros((N, paths))

    for i in range(N-1, -1, -1):
        Z = discounted_payoff[(i+1):, :] 
        if i + 1 < N:
            sum_to_remove = np.vstack([np.zeros((1, paths)), np.cumsum(martingale_increments[(i+1):, :], axis=0)])
        else:
            sum_to_remove = np.zeros((Z.shape[0], paths))
            
        vector_to_maximize = np.max(Z - sum_to_remove, axis=0)

        # Z_i: discounted payoff at current exercise time i
        Z_i = discounted_payoff[i, :]

        martingale_increment_i = np.zeros(paths)
        for j in range(N_bar):
            k = i * N_bar + j
            delta_A = discounted_asset[k+1, :] - discounted_asset[k, :] 
            indicators = get_indicator_basis_at_time_mod(asset[k, :], P)
            delta_X_all = indicators * delta_A[np.newaxis, :]  
            A_all = np.mean(delta_X_all**2, axis=1)  
            # THIS IS THE MAIN DIFFERENCE : SUBTRACTING Z_i
            b_all = np.mean(delta_X_all * (vector_to_maximize - Z_i)[np.newaxis, :], axis=1) 
            # Regolarization
            A_all_reg = A_all + 1e-8 
            alpha_all = b_all / A_all_reg
            coefficients[i, j, :] = alpha_all
            martingale_increment_i += alpha_all @ delta_X_all   
        martingale_increments[i, :] = martingale_increment_i

    martingale = np.cumsum(martingale_increments, axis=0) 
    return coefficients, martingale

def upper_bound_alfonsi_indicator_mod(asset, asset_pricing, discounted_asset, discounted_asset_pricing, discounted_payoff, discounted_payoff_pricing, paths, pricing_paths, N, N_bar, P):
    #Training
    coefficients, martingale_train = computing_coefficients_martingale_alfonsi(N, N_bar, P, discounted_asset, discounted_payoff, paths, asset)
    max_path_train = np.max(discounted_payoff - np.vstack([np.zeros((1, paths)), martingale_train]), axis=0)
    option_price_upper = np.mean(max_path_train)
    # Pricing
    martingale_pricing = alfonsi_martingale(N, N_bar, P, discounted_asset_pricing, coefficients, pricing_paths, asset_pricing)
    max_per_path = np.max(discounted_payoff_pricing - np.vstack([np.zeros((1, pricing_paths)), martingale_pricing]), axis=0)
    option_price_upper_hat = np.mean(max_per_path)
    uncertainty = np.std(max_per_path, ddof=1) / np.sqrt(pricing_paths)

    return option_price_upper_hat, option_price_upper, uncertainty, coefficients 