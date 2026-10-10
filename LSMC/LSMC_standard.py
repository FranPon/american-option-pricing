import numpy as np
from basis import poly_basis_ram
from standardization_ridge import fit_regression_coefficients
from math import comb

def ls_train_multi_standard(payoff, strike_price, exercise_dates, interval, interest_rate,
                            train_set,  max_deg, ridge_alpha):
    nb_stocks = train_set.shape[1]
    nb_basis = comb(nb_stocks + max_deg, max_deg)
    cashflow = payoff(train_set[-1], strike_price) * np.exp(-interest_rate * exercise_dates * interval)  
    coefficients = np.zeros((exercise_dates - 1, nb_basis)) 

    for t in range(exercise_dates - 1, 0, -1):
        payoff_at_t = payoff(train_set[t], strike_price) * np.exp(-interest_rate * t * interval)  
        basis_at_t = poly_basis_ram(train_set[t], max_deg) 
        itm = payoff_at_t > 0  
        Y_itm = cashflow[itm] 
        X_itm_vander = basis_at_t[:,itm]
        coefficients[t-1] = fit_regression_coefficients( X_itm_vander.T, Y_itm, ridge_alpha ) # shape: (nneurons,)
        continuation_value = np.einsum("ij,i->j", basis_at_t, coefficients[t-1]) # Continuation value for all paths 
        exercise = (payoff_at_t > continuation_value) & (itm) # Exercise condition at time t
        cashflow = np.where(exercise, payoff_at_t, cashflow) 

    return coefficients

def american_LSMC_standard(payoff, strike_price, interest_rate, maturity, interval, 
                  train_set, test_set, npaths_pricing, max_degree,
                  ridge_alpha):

    # Training phase
    exercise_dates = int( round(maturity / interval) ) 
    coefficients = ls_train_multi_standard(payoff, strike_price, exercise_dates, interval, interest_rate, 
                                           train_set,  max_degree, ridge_alpha)

    # Pricing phase
    exercise_payoff = np.zeros(npaths_pricing) # Optimal exercise time for each path, initialized to N (never early-exercised)
    has_exercised = np.zeros(npaths_pricing, dtype=bool) # Tracking if the option has been exercised for each path

    # Computing the price
    for t in range(0, exercise_dates - 1 ):
        payoff_at_t = payoff(test_set[t + 1], strike_price) * np.exp(-interest_rate * (t + 1) * interval)  # Intrinsic value at time t discounted to time 0
        itm = payoff_at_t > 0  # Indices of ITM paths at time t
        basis_at_t  = poly_basis_ram(test_set[t + 1], max_degree)
        continuation_value_pricing = np.einsum("ij,i->j", basis_at_t, coefficients[t])
        exercise_condition = (payoff_at_t > continuation_value_pricing) & itm & (~has_exercised)
        exercise_payoff[exercise_condition] = payoff_at_t[exercise_condition]
        has_exercised[exercise_condition] = True 

    final_payoff = payoff(test_set[-1], strike_price) * np.exp(-interest_rate * exercise_dates * interval)
    exercise_payoff[~has_exercised] = final_payoff[~has_exercised]

    option_price = np.mean(exercise_payoff) 
    uncertainty = np.std(exercise_payoff, ddof=1) / np.sqrt(npaths_pricing)

    immediate_exercise = payoff(test_set[0], strike_price)[0] # Immediate exercise value at time 0 is equal for all paths
    if immediate_exercise >= option_price:
        option_price = immediate_exercise
        uncertainty = 0.0

    return option_price, uncertainty