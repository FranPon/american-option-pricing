from math import comb
from basis import poly_basis_ram
from standardization_ridge import fit_regression_coefficients
import numpy as np



def american_HK_standard(payoff, strike_price, interest_rate, exercise_dates, interval, 
                         train_set, price_set, max_deg, ridge_alpha): 

    nb_stocks = train_set.shape[1]
    nb_basis = comb(nb_stocks + max_deg, max_deg)
    cashflow = payoff(train_set[-1], strike_price) * np.exp(-interest_rate * exercise_dates * interval)  
    coefficients = np.zeros((exercise_dates - 1, nb_basis)) 

    # Training phase: regression over all paths
    for t in range(exercise_dates - 2, -1, -1):
        payoff_at_t = payoff(train_set[t+1], strike_price) * np.exp(-interest_rate * (t + 1) * interval) 
        basis_at_t = poly_basis_ram(train_set[t + 1], max_deg)
        coefficients[t] = fit_regression_coefficients( basis_at_t.T, cashflow, ridge_alpha)
        cont_value_train = np.einsum("ij,i->j", basis_at_t, coefficients[t]) 
        exercise =  payoff_at_t > cont_value_train
        cashflow[exercise] = payoff_at_t[exercise]

    # Pricing phase
    npaths_pricing = price_set.shape[2]
    exercise_payoff = np.zeros(npaths_pricing)
    has_exercised = np.zeros(npaths_pricing, dtype=bool)
    cont_value = np.zeros((exercise_dates, npaths_pricing)) # E_0[V_1], E_1[V_2], ..., E_{N-1}[V_N]
    value_process = np.zeros((exercise_dates + 1, npaths_pricing)) # V_0, ..., V_N
    for t in range(0, exercise_dates - 1 ):
        basis_at_t = poly_basis_ram(price_set[t+1], max_deg)
        payoff_at_t = payoff(price_set[t+1], strike_price) * np.exp(-interest_rate * (t + 1) * interval) # Intrinsic value at time t discounted to time 0
        cont_value[t + 1] = np.einsum("ij,i->j", basis_at_t, coefficients[t]) # Approximation of the continuation value for all paths
        exercise_condition = (payoff_at_t > cont_value[t+1]) & (~has_exercised)
        exercise_payoff[exercise_condition] = payoff_at_t[exercise_condition]
        has_exercised[exercise_condition] = True 
        value_process[t + 1, : ]= np.maximum( payoff_at_t, cont_value[t+ 1,:])
    cont_value[0, :] = cashflow.mean() 
    #cont_value[0,:] = value_process[1,:].mean()

    # Lower bound regressing over all paths
    final_payoff = payoff(price_set[-1], strike_price) * np.exp(-interest_rate * (exercise_dates ) * interval)
    exercise_payoff[~has_exercised] = final_payoff[~has_exercised]
    lower_mc = np.mean(exercise_payoff)
    immediate_exercise = payoff(price_set[0], strike_price)[0] 

    if immediate_exercise >= lower_mc:
        lower_bound = immediate_exercise
        uncertainty_lower = 0.0
    else:
        lower_bound = lower_mc
        uncertainty_lower = ( np.std(exercise_payoff, ddof=1) / np.sqrt(npaths_pricing))   

    
    value_process[-1,:] = final_payoff
    value_process[0,:] = lower_bound

    # Computing the martingale
    martingale = np.zeros( (exercise_dates + 1, npaths_pricing))

    for i in range(exercise_dates):
        martingale[i + 1] = martingale[i] + value_process[i + 1] - cont_value[i]

    intrinsic_value_price = np.zeros((exercise_dates + 1, npaths_pricing))
    for i in range(exercise_dates + 1):
        intrinsic_value_price[i] = payoff(price_set[i], strike_price) * np.exp(-interest_rate * (i) * interval)

    maximizing = np.max( intrinsic_value_price - martingale, axis=0 )

    upper_bound = np.mean(maximizing)
    uncertainty_upper = np.std(maximizing, ddof=1) / np.sqrt(npaths_pricing)

    return upper_bound, uncertainty_upper, lower_bound, uncertainty_lower
