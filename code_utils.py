import numpy as np
import matplotlib.pyplot as plt

# Utility function 
# Simulation of the underlying asset price using Geometric Brownian Motion (GBM)
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
        
def transformation(X):
    trans = np.log(X)
    std = np.std(X, axis=1, keepdims=True)
    std[0] = 1.0
    mean = np.mean(trans, axis=1, keepdims=True)
    return (trans - mean) / std

# This outputs the  estimated standard deviation / sqrt(indipendent random variables) of the Monte Carlo estimator, which is the uncertainty of the estimator.
def antithetic_variates(exercise_payoff, b):
    indipendent_variables = (exercise_payoff[:b//2] + exercise_payoff[b//2:] )/2
    uncertainty = np.std(indipendent_variables, ddof=1) / np.sqrt(b//2)
    return uncertainty


def lower_bound_ls(K, S0, r, q, T, dt, sigma, train_set, test_set, train_log, test_log, discount_factors, npaths_training, npaths_pricing, max_deg, basis_train, basis_pricing, seed):
    mu = r - q
    N = int(T / dt)
    # Precomputing the intrinsic value at each time step discounted to time 0 for the training set
    intrinsic_value_train = np.maximum(0, K - S0*train_set) * discount_factors[:,np.newaxis] 

    # Training phase : Backward induction compute the coefficients
    value = np.empty((N-1, npaths_training)) # Option value at time t for each path, initialized to 0 at maturity
    value[-1,:] = intrinsic_value_train[-1,:] # Intrinsic value at time 0, discounted to time 0
    
    train_deg = np.zeros(N-1, dtype=int) # Actual degree used for regression at each time step
    coefficients = np.empty((N-1, max_deg + 1)) 
    continuation_value = np.zeros(( N-1, npaths_training)) 

    itm_train = intrinsic_value_train > 0 # Indices of ITM paths at each time step, discounted to time 0

    for t in range(N - 1, 0, -1):
        itm = itm_train[t,:] # Indices of ITM paths at time t, discounted to time 0
        itm_paths = np.sum(itm) # Number of in the money paths at time t

        # Setting itm True for all paths to see the effect of regression on the continuation value estimation
        #itm = np.ones_like(payoff_t, dtype=bool) # All paths are considered in the regression, even the OTM ones
        
        # Regression to estimate continuation value, hence to compute the coefficient 
        train_deg[t-1] = min(max_deg, int(itm_paths / 100) ) # Degree of the polynomial basis for regression

        if (train_deg[t-1] > 0) & (itm_paths > train_deg[t-1]):
            Y_itm = value[t-1, itm] 
            vander_full = basis_train[t-1, :, :train_deg[t-1]+1]
            X_itm_vander = basis_train[t-1, itm , :train_deg[t-1]+1]
            coefficients[t-1, :train_deg[t-1]+1], _, _, _ = np.linalg.lstsq(X_itm_vander, Y_itm, rcond=None) # Coefficients of the polynomial regression
            
            continuation_value[t-1,:] = vander_full @ coefficients[t-1, :train_deg[t-1]+1]
            # Determine the option value at time t
            exercise = (intrinsic_value_train[t,:] > continuation_value[t-1,:]) & (itm) # Exercise condition at time t
            # The option is early-exercised if the payoff is positive and greater than the continuation value!
            value[t-2, : ] = np.where(exercise, intrinsic_value_train[t,:], value[t-1,:]) # If the option is exercised, the value is the payoff, otherwise it's the continuation value
            #value[exercise] = intrinsic_value_train[t,exercise] # If the option is exercised, the value is the payoff
    
    coeffs_train_dict = coefficients
    contin_val_train_dict = continuation_value	
    value_train_dict = value
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
    
    option_price = np.mean(exercise_payoff)

    immediate_exercise = max(0, K - S0) # Immediate exercise value at time 0
    option_price = max(option_price, immediate_exercise)
    uncertainty = antithetic_variates(exercise_payoff, npaths_pricing)
    
    return option_price, uncertainty, coeffs_train_dict, contin_val_train_dict, value_train_dict