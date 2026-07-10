import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Utility function 

# Simulation of the underlying asset price using Geometric Brownian Motion (GBM) using antithetic variates 
def simulate_gbm( mu, sigma, T, dt, b, seed):
    np.random.seed(seed)
    N = int( round(T / dt) )  # Number of time steps
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

# Determining mean and standard deviation for log-standardization of the input data X
def fit_log_standardizer(X, eps=1e-12):
    lg = np.log(X)
    mean = lg.mean(axis=1, keepdims=True)
    std = lg.std(axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)     # guard: t=0 deterministico
    return mean, std

def apply_log_standardizer(X, stats):
    mean, std = stats
    return (np.log(X) - mean) / std

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
    N = int( round(T / dt) ) 
    
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

def ls_train_alfonsi(K, N, N_bar, P, asset, discounted_asset, basis, discount_factors, coefficients, paths, exercise_indices, max_deg):
    # Determining the discounted paygoff
    asset_exercise = asset[list(exercise_indices),:] # Shape (N, npaths_training)
    intrinsic_value = np.maximum(0, K - asset_exercise) * discount_factors[:,np.newaxis] 
    # Determining the martingale using the Alfonsi scheme for the modified continuation value
    martingale = alfonsi_martingale(N, N_bar, P, discounted_asset, coefficients, paths, asset ) # Shape (N, paths)

    # Training phase : Backward induction compute the coefficients
    value = intrinsic_value[-1,:] - martingale[-1, :] # Modified payoff 
    
    train_deg = np.zeros(N-1, dtype=int) # Actual degree used for regression at each time step
    reg_coefficients = np.zeros((N - 1, max_deg + 1)) 
    itm = intrinsic_value > 0 # We are not modified the ITM paths since we just modified the continuation value bc of (7)

    for t in range(N - 1, 0, -1):
        itm_paths = np.sum(itm[t]) # Number of in the money paths at time t
        train_deg[t-1] = min(max_deg, int(itm_paths / 100) ) 
        # " Better to use (7) instead of (4)"
        target = value + martingale[t-1,:]

        if (train_deg[t-1] > 0) & (itm_paths > train_deg[t-1]):
            Y_itm = target[itm[t]]
            vander_full = basis[t-1, :, :train_deg[t-1]+1]
            X_itm_vander = basis[t-1, itm[t] , :train_deg[t-1]+1]
            reg_coefficients[t-1, :train_deg[t-1]+1], _, _, _ = np.linalg.lstsq(X_itm_vander, Y_itm, rcond=None) 
            continuation = vander_full @ reg_coefficients[t-1, :train_deg[t-1]+1]
            
            exercise = (intrinsic_value[t,:]  > continuation) & itm[t]
            
            value = np.where(exercise, intrinsic_value[t,:] - martingale[t-1, :], value)

    return reg_coefficients, train_deg


def ls_apply_alfonsi(K, N, N_bar, P, asset, discounted_asset, basis, discount_factors, coefficients_alfonsi, regression_coefs, paths, exercise_indices, max_deg, train_deg):

    asset_exercise = asset[list(exercise_indices), :]
    intrinsic_value = (np.maximum(0, K - asset_exercise)* discount_factors[:, np.newaxis])

    martingale = alfonsi_martingale( N, N_bar, P, discounted_asset, coefficients_alfonsi, paths, asset)
    matrix_exercise_time = np.zeros((N, paths), dtype=bool)
    has_exercised = np.zeros(paths, dtype=bool)

    itm = intrinsic_value > 0
    for t in range(0, N-1):
        itm_paths = np.sum(itm[t+1]) # Number of in the money paths at time t
        if (train_deg[t] > 0) & (itm_paths > train_deg[t]):
            vander = basis[t,:,:train_deg[t]+1]
            continuation = vander @ regression_coefs[t,:train_deg[t]+1]
            exercise_condition = (intrinsic_value[t+1, :] > continuation) & (~has_exercised) & itm[t+1]
            matrix_exercise_time[t, exercise_condition] = True
            has_exercised |= exercise_condition

    matrix_exercise_time[-1, ~has_exercised] = True

    return matrix_exercise_time, martingale

def activation(x):
	return x / (1.0 + np.exp(-x))          # SiLU

# Observe that the seed is used only is weights are not provided, otherwise the provided weights are used
def random_basis(S, nneurons, sigma, seed, weights=None):
    # S: (N_bar, paths); basis (1, csi_1, ..., csi_nneurons), csi_n(x) = activation(A_n x + b_n).
    if weights is None:
        np.random.seed(seed)  # For reproducibility
        A = sigma * np.random.standard_normal(nneurons)
        b = sigma * np.random.standard_normal(nneurons)
    else:
        A, b = weights 
    activ = activation(A[:, None, None] * S[None, :, :] + b[:, None, None])  # (nneurons, N_bar, paths)
    const = np.ones((1, S.shape[0], S.shape[1]))
    basis = np.concatenate([const, activ], axis=0)                          # (nneurons+1, N_bar, paths)
    return basis, (A, b)

def ls_train_alfonsi_rnn(K, N, N_bar, P, asset, discounted_asset, basis, discount_factors, coefficients, paths, exercise_indices, max_deg, sigma_rnn_mart, weights, stats_store, seed):
    # Determining the discounted paygoff
    asset_exercise = asset[list(exercise_indices),:] # Shape (N, npaths_training)
    intrinsic_value = np.maximum(0, K - asset_exercise) * discount_factors[:,np.newaxis] 
    # Determining the martingale using the Alfonsi scheme for the modified continuation value
    martingale = alfonsi_martingale_rnn(N, N_bar, P, discounted_asset, coefficients, paths, asset, sigma_rnn_mart, weights, stats_store, seed) # Shape (N, paths)

    # Training phase : Backward induction compute the coefficients
    value = intrinsic_value[-1,:] - martingale[-1, :] # Modified payoff 
    
    train_deg = np.zeros(N-1, dtype=int) # Actual degree used for regression at each time step
    reg_coefficients = np.zeros((N - 1, max_deg + 1)) 
    itm = intrinsic_value > 0 # We are not modified the ITM paths since we just modified the continuation value bc of (7)

    for t in range(N - 1, 0, -1):
        itm_paths = np.sum(itm[t]) # Number of in the money paths at time t
        train_deg[t-1] = min(max_deg, int(itm_paths / 100) ) 
        # " Better to use (7) instead of (4)"
        target = value + martingale[t-1,:]

        if (train_deg[t-1] > 0) & (itm_paths > train_deg[t-1]):
            Y_itm = target[itm[t]]
            vander_full = basis[t-1, :, :train_deg[t-1]+1]
            X_itm_vander = basis[t-1, itm[t] , :train_deg[t-1]+1]
            reg_coefficients[t-1, :train_deg[t-1]+1], _, _, _ = np.linalg.lstsq(X_itm_vander, Y_itm, rcond=None) 
            continuation = vander_full @ reg_coefficients[t-1, :train_deg[t-1]+1]
            
            exercise = (intrinsic_value[t,:]  > continuation) & itm[t]
            
            value = np.where(exercise, intrinsic_value[t,:] - martingale[t-1, :], value)

    return reg_coefficients, train_deg


def ls_apply_alfonsi_rnn(K, N, N_bar, P, asset, discounted_asset, basis, discount_factors, coefficients_alfonsi, regression_coefs, paths, exercise_indices, train_deg, sigma_rnn_mart, weights, stats_store, seed):

    asset_exercise = asset[list(exercise_indices), :]
    intrinsic_value = (np.maximum(0, K - asset_exercise)* discount_factors[:, np.newaxis])

    martingale = alfonsi_martingale_rnn( N, N_bar, P, discounted_asset, coefficients_alfonsi, paths, asset, sigma_rnn_mart, weights, stats_store, seed)
    matrix_exercise_time = np.zeros((N, paths), dtype=bool)
    has_exercised = np.zeros(paths, dtype=bool)

    itm = intrinsic_value > 0
    for t in range(0, N-1):
        itm_paths = np.sum(itm[t+1]) # Number of in the money paths at time t
        if (train_deg[t] > 0) & (itm_paths > train_deg[t]):
            vander = basis[t,:,:train_deg[t]+1]
            continuation = vander @ regression_coefs[t,:train_deg[t]+1]
            exercise_condition = (intrinsic_value[t+1, :] > continuation) & (~has_exercised) & itm[t+1]
            matrix_exercise_time[t, exercise_condition] = True
            has_exercised |= exercise_condition

    matrix_exercise_time[-1, ~has_exercised] = True

    return matrix_exercise_time, martingale


def computing_coefficients_martingale_alfonsi_rnn(N, N_bar, nneurons, discounted_asset, discounted_payoff, paths, asset, sigma, seed):

    h = nneurons + 1
    coefficients = np.zeros((N, N_bar, h))    
    martingale_increments = np.zeros((N, paths))
    weights_store = [None] * N                 # weights per block, reused in pricing
    stats_store = [None] * N                   # stats per block, reused in pricing
    np.random.seed(seed)         
    # One draw
    A = sigma * np.random.standard_normal(nneurons)
    b = sigma * np.random.standard_normal(nneurons)
    shared_w = (A, b)
        
    for i in range(N - 1, -1, -1):
        Z = discounted_payoff[(i + 1):, :]
        if i + 1 < N:
            sum_to_remove = np.vstack([np.zeros((1, paths)), np.cumsum(martingale_increments[(i + 1):, :], axis=0)])
        else:
            sum_to_remove = np.zeros((Z.shape[0], paths))

        vector_to_maximize = np.max(Z - sum_to_remove, axis=0)
        Z_i = discounted_payoff[i, :]
        target = vector_to_maximize - Z_i                                   # (paths,), COMMON to all j

        asset_block = asset[i * N_bar:(i + 1) * N_bar, :]
        dblk = discounted_asset[i * N_bar:(i + 1) * N_bar + 1, :]
        difference_discounted_asset = dblk[1:, :] - dblk[:-1, :]            # (N_bar, paths)
        # Standardized asset block for better numerical stability
        stats_store[i] = fit_log_standardizer(asset_block)
        asset_block_std = apply_log_standardizer(asset_block, stats_store[i])	
        
        rnn_basis, w = random_basis(asset_block_std, nneurons, sigma, seed, shared_w)         # (h, N_bar, paths)
        weights_store[i] = w
        pre_X = rnn_basis * difference_discounted_asset[np.newaxis, :, :]   # (h, N_bar, paths)

        incr_i = np.zeros(paths)
        for j in range(N_bar):                                             	# the N_bar decoupled systems
            dXj = pre_X[:, j, :]                                            # (h, paths) = Delta X_{i,j}
            alpha_j = np.linalg.lstsq(dXj.T, target, rcond=None)[0]         # (h,)
            coefficients[i, j, :] = alpha_j
            incr_i += alpha_j @ dXj
        martingale_increments[i, :] = incr_i

    martingale = np.cumsum(martingale_increments, axis=0)
    return coefficients, martingale, weights_store, stats_store

# Since in this function weights are always provided, the seed is not used. However, it is kept for consistency with the other functions.
def alfonsi_martingale_rnn(N, N_bar, nneurons, discounted_asset, coefficients, paths, asset, sigma, weights_store, stats_store, seed):
    martingale_increments = np.zeros((N, paths))
    for i in range(N - 1, -1, -1):
        asset_block = asset[i * N_bar:(i + 1) * N_bar, :]
        dblk = discounted_asset[i * N_bar:(i + 1) * N_bar + 1, :]
        difference_discounted_asset = dblk[1:, :] - dblk[:-1, :]

        # Standardized asset block for better numerical stability
        asset_block_std = apply_log_standardizer(asset_block, stats_store[i])	

        rnn_basis, _ = random_basis(asset_block_std, nneurons, sigma, seed, weights=weights_store[i])      # SAME weights as training
        pre_X = rnn_basis * difference_discounted_asset[np.newaxis, :, :]
        incr_i = np.zeros(paths)
        for j in range(N_bar):
            dXj = pre_X[:, j, :]
            incr_i += coefficients[i, j, :] @ dXj
        martingale_increments[i, :] = incr_i
    return np.cumsum(martingale_increments, axis=0)

def upper_bound_alfonsi_rnn(asset, asset_pricing, discounted_asset, discounted_asset_pricing, discounted_payoff, discounted_payoff_pricing, paths, pricing_paths, N, N_bar, nneurons, sigma, seed_weights):

    coefficients, martingale_train, weights_store, stats_store = computing_coefficients_martingale_alfonsi_rnn( N, N_bar, nneurons, discounted_asset, discounted_payoff, paths, asset, sigma, seed_weights)
    max_path_train = np.max(discounted_payoff - np.vstack([np.zeros((1, paths)), martingale_train]), axis=0)
    option_price_upper = np.mean(max_path_train)

    martingale_pricing = alfonsi_martingale_rnn( N, N_bar, nneurons, discounted_asset_pricing, coefficients, pricing_paths, asset_pricing, sigma, weights_store, stats_store, seed_weights)                           
    max_per_path = np.max(discounted_payoff_pricing - np.vstack([np.zeros((1, pricing_paths)), martingale_pricing]), axis=0)
    option_price_upper_hat = np.mean(max_per_path)
    uncertainty = np.std(max_per_path, ddof=1) / np.sqrt(pricing_paths)

    return option_price_upper_hat, option_price_upper, uncertainty, coefficients, weights_store

# LONGSTAFF-SCHWARTZ METHOD FOR AMERICAN PUT OPTIONS USING POLYNOMIAL BASIS FUNCTIONS
def lower_bound_american_put_ls(K, S0, T ,dt, r, q, sigma, seed, npaths_train, npaths_pricing, max_deg):
    mu = r - q  # drift risk-neutral
    N = int(round(T / dt))  

    # Paths simulation 
    train_set = simulate_gbm(mu, sigma, T, dt, npaths_train, seed=seed)
    test_set = simulate_gbm( mu, sigma, T, dt, npaths_pricing, seed = seed + 1)

    # Applying transformation
    stats = fit_log_standardizer(train_set)          		# fit only on training
    train_log = apply_log_standardizer(train_set, stats)
    test_log = apply_log_standardizer(test_set,  stats) 

    # Discount factors for each time step
    discount_factors = np.exp(-r * np.arange(0, N + 1) * dt) 

    # Computing the basis for the regression 
    basis_train = np.ones((N-1,npaths_train, max_deg + 1))	
    basis_pricing = np.ones((N-1, npaths_pricing, max_deg + 1))
    for d in range(1, max_deg + 1):
        basis_train[:, :, d] = train_log[1:N,:] * basis_train[:, :, d-1]
        basis_pricing[:, :, d] = test_log[1:N,:] * basis_pricing[:, :, d-1]

    price_estimate, uncertainty_estimate, _, _, _ , _= lower_bound_ls(K, S0, T, dt, train_set, test_set, discount_factors, npaths_train, npaths_pricing, max_deg, basis_train, basis_pricing)
    return price_estimate , uncertainty_estimate

# ROGERS/HAUGH/KOGAN METHOD FOR AMERICAN PUT OPTIONS USING POLYNOMIAL BASIS FUNCTIONS
# Upper bound using Rogers/Haugh/Kogan method given the basis
def upper_bound_rhg(K, S0, T, dt, N, train_ls, test_ls, train_reg, price_set, discount_factors, npaths_train, npaths_pricing, max_deg, basis_train_ls, basis_test_ls, basis_train_reg, basis_price):

    lower_bound, _, coefficients, _, _, _  = lower_bound_ls(K, S0, T, dt, train_ls, test_ls, discount_factors, npaths_train, npaths_pricing, max_deg, basis_train_ls, basis_test_ls)
    price = lower_bound
    coefficients_cont_value_itm = coefficients # Coefficients for estimating the continuation value
    Z_reg = np.maximum(0, K - S0 * train_reg) * discount_factors[:, np.newaxis]
    V_tilde_reg = np.zeros((N+1, npaths_train))
    V_tilde_reg[0,:] = price
    V_tilde_reg[-1,:] = Z_reg[-1,:]
    coefficients_reg = np.zeros((N-1, max_deg + 1 ))
    Q_tilde_reg = np.zeros((N-1, npaths_train)) # E_1[V_2], ..., E_{N-1}[V_N]
    for t in range(N - 1, 0, -1):
        value = V_tilde_reg[t+1,:]
        vander_full = basis_train_reg[t-1, :, :] # the t-1 is given by the shape of basis_train_reg
        itm = Z_reg[t,:] > 0
        coefficients_reg[t-1,:], _, _, _ = np.linalg.lstsq(vander_full, value, rcond=None) 
        continuation_value_itm = basis_train_reg[t-1,:,:] @ coefficients_cont_value_itm[t-1,:] 
        Q_tilde_reg[t-1,:] = basis_train_reg[t-1,:,:] @ coefficients_reg[t-1,:]
        V_tilde_reg[t, : ]= np.where( itm & (Z_reg[t,:] > continuation_value_itm), Z_reg[t,:], Q_tilde_reg[t-1,:])


    # To avoid overfitting we use another Montercarlo to compute the value, the continuation value and the respective martingale
    Z = np.maximum(0, K - S0 * price_set) * discount_factors[:, np.newaxis] 
    V_tilde = np.zeros((N+1, npaths_pricing)) # V_0, ..., V_N
    V_tilde[0,:] = price
    V_tilde[-1,:] = Z[-1,:]
    Q_tilde = np.zeros((N, npaths_pricing)) # E_0[V_1], E_1[V_2], ..., E_{N-1}[V_N]
    
    for t in range(N-1, 0, -1):
        vander_full = basis_price[t-1, :, :] # the t-1 is given by the shape of basis_train_reg
        itm = Z[t,:] > 0
        cont_value_itm = basis_price[t-1,:,:] @ coefficients_cont_value_itm[t-1,:] 
        Q_tilde[t,:] = basis_price[t-1,:,:] @ coefficients_reg[t-1,:]
        V_tilde[t, : ]= np.where( itm & (Z[t,:] > cont_value_itm), Z[t,:], Q_tilde[t,:])
    Q_tilde[0,:] = V_tilde[1,:].mean()

    # Computing the martingale
    martingale = np.zeros((N+1, npaths_pricing)) 
    martingale[0,:] = price
    for i in range(N):
        martingale[i+1,:] = martingale[i,:] + V_tilde[i+ 1 ,:] - Q_tilde[i,:]

    # Second Montecarlo : determining the coefficient for the estimating E_t[ V_{t+1} - V_t ] 

    # Upper bound
    maximizing = price + np.max(Z - martingale, axis=0)
    antithetic_sum = (maximizing[:npaths_pricing//2] + maximizing[npaths_pricing//2:]) / 2
    option_price = np.mean(antithetic_sum)

    upper_bound = option_price
    uncertainty= np.std(antithetic_sum, ddof=1) / np.sqrt(npaths_pricing//2)

    return upper_bound, uncertainty

def upper_bound_american_put_rhg(K, S0, T, dt, r, q, sigma, seed, npaths_train, npaths_pricing, max_deg):
    mu = r - q  # drift risk-neutral
    N = int(round(T / dt)) # Number of steps
    
    # Paths simulation 
    train_ls = simulate_gbm(mu, sigma, T, dt, npaths_train, seed=seed)
    test_ls = simulate_gbm(mu, sigma, T, dt, npaths_pricing, seed=seed + 1)

    train_reg = simulate_gbm(mu, sigma, T, dt, npaths_train, seed = seed + 2)
    price_set = simulate_gbm(mu, sigma, T, dt, npaths_pricing, seed = seed + 3)

    # Applying transformation
    stats = fit_log_standardizer(train_ls)          
    train_ls_log = apply_log_standardizer(train_ls, stats)
    test_ls_log = apply_log_standardizer(test_ls, stats)

    stats_reg = fit_log_standardizer(train_reg)
    train_reg_log = apply_log_standardizer(train_reg, stats_reg)
    price_log  = apply_log_standardizer(price_set, stats_reg)

    # Discount factors for each time step
    discount_factors = np.exp(-r * np.arange(0, N + 1) * dt) 

    # Precomputing the basis for the regression since after transformation the input doesn't depend on S0
    basis_train_ls= np.ones((N-1,npaths_train, max_deg + 1))
    basis_test_ls = np.ones((N-1, npaths_pricing, max_deg + 1)) # uniquely used for determining the correct lower price	
    basis_train_reg = np.ones((N-1, npaths_train, max_deg + 1))
    basis_price = np.ones((N-1, npaths_pricing, max_deg + 1))
    for d in range(1, max_deg + 1):
        basis_train_ls[:, :, d] = train_ls_log[1:N,:] * basis_train_ls[:, :, d-1]
        basis_test_ls[:, :, d] = test_ls_log[1:N,:] * basis_test_ls[:, :, d-1]
        basis_train_reg[:, :, d] = train_reg_log[1:N,:] * basis_train_reg[:, :, d-1]
        basis_price[:,:,d] = price_log[1:N,:] * basis_price[:,:,d-1]

    price_estimate, uncertainty_estimate = upper_bound_rhg(K, S0, T, dt, N, train_ls, test_ls, train_reg, price_set, discount_factors, npaths_train, npaths_pricing, max_deg, basis_train_ls, basis_test_ls, basis_train_reg, basis_price)
    return price_estimate, uncertainty_estimate
  
# LOWER BOUND ALFONSI METHOD FOR AMERICAN PUT OPTIONS USING INDICATOR BASIS FUNCTIONS
def lower_bound_american_put_alfonsi(K, S0, T, N, N_bar, r, sigma, P, seed, paths_alfonsi, paths_ls_train, paths_ls_pricing, max_deg):
    
    # Temporal grids
    exercise_time = np.linspace(0, T, N + 1)
    exercise_indices = range(0, N_bar * N + 1, N_bar)
    dense_grid = np.linspace(0, T, N_bar * N + 1)

    discount_factor = np.exp(-r * dense_grid)
    discount_factor_exercise_time = np.exp(-r * exercise_time)

    # Step 1 : determining the martingale for the control variate using Alfonsi's scheme
    asset_alfonsi = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths_alfonsi, seed)
    discounted_asset_alfonsi = asset_alfonsi * discount_factor[:, np.newaxis]
    discounted_payoff_alfonsi= (np.maximum(K - asset_alfonsi[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])
    #Determining the martingale
    coefficients_alfonsi, _ = computing_coefficients_martingale_alfonsi(N, N_bar, P, discounted_asset_alfonsi, discounted_payoff_alfonsi, paths_alfonsi, asset_alfonsi)


    # Step 2 : Longstaff-Schwartz method to get the optimal stopping rule
    asset_ls = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths_ls_train, seed + 1)
    discounted_asset_ls = asset_ls * discount_factor[:, np.newaxis]

    stats_reg  = fit_log_standardizer(asset_ls)
    ls_log = apply_log_standardizer(asset_ls, stats_reg)
    ls_log_ex = ls_log[list(exercise_indices), :]      # (N+1, paths)
    basis_ls = np.ones((N-1, paths_ls_train, max_deg + 1))
    for d in range(1, max_deg + 1):
        basis_ls[:, :, d] = ls_log_ex[1:N, :] * basis_ls[:, :, d-1]  
        
    regression_coefs, train_deg = ls_train_alfonsi(K, N, N_bar, P, asset_ls, discounted_asset_ls, basis_ls, discount_factor_exercise_time, coefficients_alfonsi, paths_ls_train, exercise_indices, max_deg)

    # Step 3 : Pricing the option using the optimal stopping rule and the martingale as control variate
    asset_pricing = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths_ls_pricing, seed + 2)
    discounted_asset_pricing = asset_pricing * discount_factor[:, np.newaxis]
    discounted_payoff_pricing = (np.maximum(K - asset_pricing[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])

    ls_price = apply_log_standardizer(asset_pricing, stats_reg)  
    ls_price_ex = ls_price[list(exercise_indices), :]

    basis_pricing = np.ones((N-1, paths_ls_pricing, max_deg + 1))
    for d in range(1, max_deg + 1):
        basis_pricing[:, :, d] = ls_price_ex[1:N, :] * basis_pricing[:, :, d-1]
        
    optimal_stopping, martingale_pricing = ls_apply_alfonsi( K, N, N_bar, P, asset_pricing, discounted_asset_pricing, basis_pricing, discount_factor_exercise_time, coefficients_alfonsi, regression_coefs, paths_ls_pricing, exercise_indices, max_deg, train_deg)

    # Determining the coefficient for the control variate
    payoff_control_variate = np.sum(discounted_payoff_pricing[1:] * optimal_stopping,axis=0) # Payoff of the control variate using the optimal stopping rule
    martingale_control_variate = np.sum(martingale_pricing * optimal_stopping,axis=0) # Martingale control variate using the optimal stopping rule
    coeff_lambda = np.sum(payoff_control_variate * martingale_control_variate) / np.sum(martingale_control_variate**2) # Coefficient for the control variate

    # Antithetic variates ( antithetic variates only for the pricing part! )
    #modified_payoff = payoff_control_variate - coeff_lambda * martingale_control_variate
    #antithetic_sum = (modified_payoff[:paths_ls_pricing//2] + modified_payoff[paths_ls_pricing//2:]) / 2
    #price_mc = np.mean(antithetic_sum)
    #std_mc = np.std(antithetic_sum, ddof=1) / np.sqrt(paths_ls_pricing//2)
    
    # No antithetic variates version
    price_mc  = np.mean(payoff_control_variate - coeff_lambda * martingale_control_variate)
    std_mc = np.std(payoff_control_variate - coeff_lambda * martingale_control_variate, ddof=1) / np.sqrt(paths_ls_pricing)
    
    # Since we are generating the paths using a variance reduction technique (antithetic variates), we need to take
    # this into account: we need to do the average of the "antithetic values", hence this is wrong option_price = np.mean(exercise_payoff)
    # antithetic_sum = (exercise_payoff[:npaths_pricing//2] + exercise_payoff[npaths_pricing//2:]) / 2
    # option_price = np.mean(antithetic_sum)

    immediate_exercise = max(K - S0, 0)
    standard_error = std_mc
    price = price_mc

    if immediate_exercise >= price_mc:
        price = immediate_exercise
        standard_error = 0.0       
    else:
        price = price_mc
        standard_error = std_mc
    
    # Results
    price_estimate = price
    uncertainty_estimate = standard_error
    return price_estimate, uncertainty_estimate

# UPPER ALFONSI METHOD FOR AMERICAN PUT OPTIONS USING INDICATOR BASIS FUNCTIONS
def upper_bound_american_put_alfonsi(K, S0, T, N, r, sigma, seed, npaths_train, npaths_pricing, N_bar, P):
    
    # Temporal grid
    exercise_time = np.linspace(0, T, N + 1)
    exercise_indices = range(0, N_bar * N + 1, N_bar)
    dense_grid = np.linspace(0, T, N_bar * N + 1)

    discount_factor = np.exp(-r * dense_grid)
    discount_factor_exercise_time = np.exp(-r * exercise_time)

    # Training paths 
    asset = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], npaths_train, seed)
    discounted_asset = asset * discount_factor[:, np.newaxis]
    discounted_payoff = (np.maximum(K - asset[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])

    # Pricing paths 
    asset_pricing = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], npaths_pricing, seed + 1)
    discounted_asset_pricing = asset_pricing * discount_factor[:, np.newaxis]
    discounted_payoff_pricing = (np.maximum(K - asset_pricing[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])
    
    price_estimate, _ , uncertainty_estimate, _ = upper_bound_alfonsi_indicator_mod( asset, asset_pricing, discounted_asset, discounted_asset_pricing, discounted_payoff, discounted_payoff_pricing, npaths_train, npaths_pricing, N, N_bar, P )
    return price_estimate , uncertainty_estimate

# LONGSTAFF-SCHWARTZ METHOD FOR AMERICAN PUT OPTIONS USING RANDOMIZED NEURAL NETWORK BASIS FUNCTIONS
def rnn_lower_bound_american_put_ls(K, S0, T, dt, r, q, sigma, seed, npaths_train, npaths_pricing, nneurons, sigma_basis, seed_weights):
	N = int(round(T / dt) )	# Number of steps
	mu = r - q  # drift risk-neutral

	# Paths simulation 
	train_set = simulate_gbm(mu, sigma, T, dt, npaths_train, seed=seed)
	test_set = simulate_gbm( mu, sigma, T, dt, npaths_pricing, seed = seed + 1)

	# Applying transformation
	stats = fit_log_standardizer(train_set)          
	train_log = apply_log_standardizer(train_set, stats)
	test_log = apply_log_standardizer(test_set,  stats) 
	# Discount factors for each time step
	discount_factors = np.exp(-r * np.arange(0, N + 1) * dt) 

	# Precomputing the basis for the regression since after transformation the input doesn't depend on S0
	# Sigma in this case represents the standard deviation of the random weights 
	basis_train_mod, weights_train = random_basis(train_log[1:N,:], nneurons, sigma_basis, seed_weights) 
	basis_pricing_mod, _ = random_basis(test_log[1:N,:], nneurons, sigma_basis, seed_weights, weights=weights_train)

	basis_train = basis_train_mod.transpose(1, 2, 0)
	basis_pricing = basis_pricing_mod.transpose(1, 2, 0)

	price_estimate, uncertainty_estimate, _, _, _ , _= lower_bound_ls(K, S0, T, dt, train_set, test_set, discount_factors, npaths_train, npaths_pricing, nneurons, basis_train, basis_pricing)
	return price_estimate, uncertainty_estimate

# ROGERS/HAUGH/KOGAN METHOD FOR AMERICAN PUT OPTIONS USING RANDOMIZED NEURAL NETWORK BASIS FUNCTIONS
def rnn_upper_bound_american_put_rhg(K, S0, T, dt, r, q, sigma, seed, npaths_train, npaths_pricing, nneurons, sigma_basis, seed_weights):
	N = int(round(T / dt) )	# Number of steps
	mu = r - q
	# Paths simulation : The first two Montecarlo are merely used for determining the lower bound
	train_ls = simulate_gbm(mu, sigma, T, dt, npaths_train, seed=seed)
	test_ls = simulate_gbm(mu, sigma, T, dt, npaths_pricing, seed=seed + 1)

	train_reg = simulate_gbm(mu, sigma, T, dt, npaths_train, seed = seed + 2)
	price_set = simulate_gbm(mu, sigma, T, dt, npaths_pricing, seed = seed + 3)

	# Applying transformation
	stats = fit_log_standardizer(train_ls)          
	train_ls_log = apply_log_standardizer(train_ls, stats)
	test_ls_log = apply_log_standardizer(test_ls, stats)

	stats_reg = fit_log_standardizer(train_reg)
	train_reg_log = apply_log_standardizer(train_reg, stats_reg)
	price_log = apply_log_standardizer(price_set, stats_reg)

	# Discount factors for each time step
	discount_factors = np.exp(-r * np.arange(0, N + 1) * dt) 

	# Precomputing the basis for the regression since after transformation the input doesn't depend on S0
	basis_train_ls_mod, weights_train = random_basis(train_ls_log[1:N,:], nneurons, sigma_basis, seed=seed_weights)
	basis_test_ls_mod, _ = random_basis(test_ls_log[1:N,:], nneurons, sigma_basis, seed=seed_weights, weights=weights_train)

	basis_train_reg_mod, _ = random_basis(train_reg_log[1:N,:], nneurons, sigma_basis, seed=seed_weights, weights=weights_train)
	basis_pricing_mod, _ = random_basis(price_log[1:N,:], nneurons, sigma_basis, seed=seed_weights, weights=weights_train)

	# Adjusting the shape of the basis 
	basis_train_ls= basis_train_ls_mod.transpose(1, 2, 0)
	basis_test_ls = basis_test_ls_mod.transpose(1, 2, 0)

	basis_train_reg = basis_train_reg_mod.transpose(1, 2, 0)
	basis_price = basis_pricing_mod.transpose(1, 2, 0)

	price_estimate, uncertainty_estimate = upper_bound_rhg(K, S0, T, dt, N, train_ls, test_ls, train_reg, price_set, discount_factors, npaths_train, npaths_pricing, nneurons, basis_train_ls, basis_test_ls, basis_train_reg, basis_price)
	return price_estimate, uncertainty_estimate

# LOWER BOUND ALFONSI METHOD FOR AMERICAN PUT OPTIONS USING RANDOMIZED NEURAL NETWORK BASIS FUNCTIONS
def rnn_lower_bound_american_put_alfonsi(K, S0, N, N_bar, r, sigma, T, seed, paths_alfonsi, paths_ls_train, paths_ls_pricing, nneurons_mart, nneurons_reg, sigma_rnn_mart, sigma_rnn_reg, seed_weights):
    # Temporal grids
    exercise_time = np.linspace(0, T, N + 1)
    exercise_indices = range(0, N_bar * N + 1, N_bar)
    dense_grid = np.linspace(0, T, N_bar * N + 1)

    discount_factor = np.exp(-r * dense_grid)
    discount_factor_exercise_time = np.exp(-r * exercise_time)

    # Step 1 : determining the martingale for the control variate using Alfonsi's scheme
    asset_alfonsi = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths_alfonsi, seed)
    discounted_asset_alfonsi = asset_alfonsi * discount_factor[:, np.newaxis]
    discounted_payoff_alfonsi= (np.maximum(K - asset_alfonsi[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])
    
    coefficients_alfonsi, _ , weights_mart, stats_mart = computing_coefficients_martingale_alfonsi_rnn(N, N_bar, nneurons_mart, discounted_asset_alfonsi, discounted_payoff_alfonsi, paths_alfonsi, asset_alfonsi, sigma_rnn_mart, seed_weights)

    # Step 2 : Longstaff-Schwartz method to get the optimal stopping rule
    asset_ls = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths_ls_train, seed + 1)
    discounted_asset_ls = asset_ls * discount_factor[:, np.newaxis]

    stats_reg  = fit_log_standardizer(asset_ls)
    ls_log = apply_log_standardizer(asset_ls, stats_reg)
    ls_log_ex = ls_log[list(exercise_indices), :]      # (N+1, paths)
    
    basis_ls = np.ones((N-1, paths_ls_train, nneurons_reg + 1))
    basis_ls_mod, weights_reg = random_basis(ls_log_ex[1:N,:], nneurons_reg, sigma_rnn_reg, seed_weights + 1)
    basis_ls = basis_ls_mod.transpose(1, 2, 0)


    regression_coefs, train_deg = ls_train_alfonsi_rnn(K, N, N_bar, nneurons_mart, asset_ls, discounted_asset_ls, basis_ls, discount_factor_exercise_time, coefficients_alfonsi, paths_ls_train, exercise_indices, nneurons_reg, sigma_rnn_mart, weights_mart, stats_mart, seed_weights)
    # Step 3 : Pricing the option using the optimal stopping rule and the martingale as control variate
    asset_pricing = S0 * simulate_gbm(r, sigma, T, dense_grid[1] - dense_grid[0], paths_ls_pricing, seed + 2)
    discounted_asset_pricing = asset_pricing * discount_factor[:, np.newaxis]
    discounted_payoff_pricing = (np.maximum(K - asset_pricing[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])

    ls_price = apply_log_standardizer(asset_pricing, stats_reg)  
    ls_price_ex = ls_price[list(exercise_indices), :]
    basis_pricing = np.ones((N-1, paths_ls_pricing, nneurons_reg + 1))
    basis_pricing_mod, _ = random_basis(ls_price_ex[1:N,:], nneurons_reg, sigma_rnn_reg, seed_weights, weights=weights_reg)
    basis_pricing = basis_pricing_mod.transpose(1, 2, 0)

        
    optimal_stopping, martingale_pricing = ls_apply_alfonsi_rnn( K, N, N_bar, nneurons_mart, asset_pricing, discounted_asset_pricing, basis_pricing, discount_factor_exercise_time, coefficients_alfonsi, regression_coefs, paths_ls_pricing, exercise_indices, train_deg, sigma_rnn_mart, weights_mart, stats_mart, seed_weights)

    # Determining the coefficient for the control variate
    payoff_control_variate = np.sum(discounted_payoff_pricing[1:] * optimal_stopping,axis=0) # Payoff of the control variate using the optimal stopping rule
    martingale_control_variate = np.sum(martingale_pricing * optimal_stopping,axis=0) # Martingale control variate using the optimal stopping rule
    coeff_lambda = np.sum(payoff_control_variate * martingale_control_variate) / np.sum(martingale_control_variate**2) # Coefficient for the control variate

    # Antithetic variates ( antithetic variates only for the pricing part! )
    modified_payoff = payoff_control_variate - coeff_lambda * martingale_control_variate
    antithetic_sum = (modified_payoff[:paths_ls_pricing//2] + modified_payoff[paths_ls_pricing//2:]) / 2
    price_mc = np.mean(antithetic_sum)
    std_mc = np.std(antithetic_sum, ddof=1) / np.sqrt(paths_ls_pricing//2)
    
    # No antithetic variates version
    #price_mc  = np.mean(payoff_control_variate - coeff_lambda * martingale_control_variate)
    #std_mc = np.std(payoff_control_variate - coeff_lambda * martingale_control_variate, ddof=1) / np.sqrt(paths_ls_pricing)
    

    # Since we are generating the paths using a variance reduction technique (antithetic variates), we need to take
    # this into account: we need to do the average of the "antithetic values", hence this is wrong option_price = np.mean(exercise_payoff)
    # antithetic_sum = (exercise_payoff[:npaths_pricing//2] + exercise_payoff[npaths_pricing//2:]) / 2
    # option_price = np.mean(antithetic_sum)

    immediate_exercise = max(K - S0, 0)
    standard_error = std_mc
    price = price_mc

    if immediate_exercise >= price_mc:
        price = immediate_exercise
        standard_error = 0.0       
    else:
        price = price_mc
        standard_error = std_mc
        
    return price, standard_error

# UPPER BOUND ALFONSI METHOD FOR AMERICAN PUT OPTIONS USING RANDOMIZED NEURAL NETWORK BASIS FUNCTIONS
def rnn_upper_bound_american_put_alfonsi(K, S0, r, sigma, sigma_basis, T, N, N_bar, nneurons, paths, pricing_paths, seed, seed_weights):
    # Temporal grid
    exercise_time = np.linspace(0, T, N + 1)
    exercise_indices = range(0, N_bar * N + 1, N_bar)
    dense_grid = np.linspace(0, T, N_bar * N + 1)
    
    discount_factor = np.exp(-r * dense_grid)
    discount_factor_exercise_time = np.exp(-r * exercise_time)

    # Training paths 
    asset = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], paths, seed)
    discounted_asset = asset * discount_factor[:, np.newaxis]
    discounted_payoff = (np.maximum(K - asset[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])

    # Pricing paths 
    asset_pricing = S0 * simulate_gbm_nv(r, sigma, T, dense_grid[1] - dense_grid[0], pricing_paths, seed + 1)
    discounted_asset_pricing = asset_pricing * discount_factor[:, np.newaxis]
    discounted_payoff_pricing = (np.maximum(K - asset_pricing[exercise_indices, :], 0) * discount_factor_exercise_time[:, np.newaxis])

    price_estimate, _, uncertainty_estimate, _,_ = upper_bound_alfonsi_rnn( asset, asset_pricing, discounted_asset, discounted_asset_pricing, discounted_payoff, discounted_payoff_pricing, paths, pricing_paths, N, N_bar, nneurons, sigma_basis, seed_weights)

    return price_estimate, uncertainty_estimate