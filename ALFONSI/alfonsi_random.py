import numpy as np
from standardization_ridge import fit_regression_coefficients
from basis import get_basis_random

# Computing martingale for control variate/upper bound using Alfonsi method knowing the coefficients
def determine_martingale(assets, fine_dt, interest_rate, dividend, exercise_dates, subintervals, coefficients, activation, A, b, stats):

    nb_stocks, paths  = assets.shape[1], assets.shape[2]

    martingale_increments = np.zeros((exercise_dates, paths))
    for t in range(exercise_dates - 1, -1, -1):
        increment_t = np.zeros(paths)

        for j in range(subintervals):
            q = t * subintervals + j  # indice sulla griglia fine

            H,_  = get_basis_random(assets[q], activation, A, b, stats[q]) # H: (nneurons, npaths_training)

            G = (assets[q + 1] * np.exp((dividend - interest_rate) * fine_dt * (q + 1))
            - assets[q] * np.exp((dividend - interest_rate) * fine_dt * q))  # G: (nb_stocks, npaths_training)

            alpha = coefficients[t, j].reshape(H.shape[0], nb_stocks)

            increment_t += np.einsum("hp,hd,dp->p", H, alpha, G, optimize=False)

        martingale_increments[t] = increment_t
    martingale = np.cumsum(martingale_increments, axis=0)
    return martingale


def computing_alfonsi_martingale_ram( exercise_dates, subintervals, interest_rate, dividend, interval_exercise, fine_dt,
                                     strike_price, payoff, activation, asset_exercise, A, b, assets_fine, ridge_alpha):
    nb_stocks, npaths_training = assets_fine.shape[1], assets_fine.shape[2]
    # Assets shape: (maturity * intervals + 1, nb_stocks, paths)
    nneurons = A.shape[0] + 1 
    coefficients = np.zeros((exercise_dates, subintervals, nneurons * nb_stocks)) 
    martingale_increments = np.zeros((exercise_dates, npaths_training))

    stats_at_t = [None] * (exercise_dates * subintervals) 
    for t in range(exercise_dates - 1, -1, -1):
        Z = np.stack([
            payoff(asset_exercise[k], strike_price)
            * np.exp(-interest_rate * interval_exercise * k)
            for k in range(t + 1, exercise_dates + 1)
        ], axis=0)

        if t + 1 < exercise_dates:
            sum_to_remove = np.vstack([
                np.zeros((1, npaths_training)),
                np.cumsum(martingale_increments[t + 1:], axis=0),
            ])
        else:
            sum_to_remove = np.zeros((1, npaths_training))

        vector_to_maximize = np.max(Z - sum_to_remove, axis=0)
        #Z_i = intrinsic_value[i, :]
        #target = vector_to_maximize - Z_i 
        target = vector_to_maximize

        incr_t = np.zeros(npaths_training)
        for j in range(subintervals):
            q = t * subintervals + j  # indice sulla griglia fine

            H, stats_at_t[q] = get_basis_random(assets_fine[q], activation, A, b)  # H: (nneurons, npaths_training)

            G = (assets_fine[q + 1] * np.exp((dividend - interest_rate) * fine_dt * (q + 1))
            - assets_fine[q] * np.exp((dividend - interest_rate) * fine_dt * q))  # G: (nb_stocks, npaths_training)

            H_regression = H.astype(np.float32, copy=False)
            G_regression = G.astype(np.float32, copy=False)
            target_regression = np.asarray(target, dtype=np.float32)
            dXj = ( H_regression.T[:, :, None] * G_regression.T[:, None, :]).reshape( npaths_training, nneurons * nb_stocks)

            if t == 0 and j == 0:
                c = H[:, 0]
                beta = np.linalg.lstsq(G.T, target, rcond=None)[0]
                alpha_j = (np.outer(c, beta) / np.dot(c, c)).reshape(-1)
            else:
                alpha_j = fit_regression_coefficients(dXj, target_regression, ridge_alpha)

            alpha_j_regression = np.asarray( alpha_j, dtype=dXj.dtype)
            coefficients[t, j] = alpha_j_regression
            incr_t += dXj @ alpha_j_regression
            del dXj
            del H_regression
            del G_regression
            del target_regression
        martingale_increments[t] = incr_t
        

    martingale = np.cumsum(martingale_increments, axis=0)

    return coefficients, martingale, stats_at_t


# ALFONSI PRIMAL DUAL
def american_AL_PRIMAL_DUAL(payoff, strike_price, activation, interest_rate, dividend, maturity, exercise_dates, 
                       subintervals, martingale_set, policy_set, price_set, nneurons_mart, nneurons_reg,
                        std_basis_mart, std_basis_reg, basis_seed, ridge_alpha):
    
    fine_dt = maturity / (exercise_dates * subintervals)
    interval_exercise = maturity / exercise_dates

    nb_stocks = martingale_set.shape[1]

    np.random.seed(basis_seed)
    A_mart = ( std_basis_mart * np.random.randn(nneurons_mart - 1, nb_stocks) ).astype(np.float32)
    b_mart = ( std_basis_mart * np.random.randn(nneurons_mart - 1) ).astype(np.float32)
    # Determine the martingale for the control variate using Alfonsi's scheme
    martingale_set_exercise = martingale_set[::subintervals]
    coefficients_martingale, _, stats_at_t_mart = computing_alfonsi_martingale_ram(exercise_dates, subintervals, interest_rate, dividend, interval_exercise, fine_dt,
                                     strike_price, payoff, activation, martingale_set_exercise, A_mart, b_mart, martingale_set, ridge_alpha)

    # Determine the optimal stopping rule using Longstaff-Schwartz method

    martingale_regression = determine_martingale(policy_set, fine_dt, interest_rate, dividend, exercise_dates, subintervals, coefficients_martingale, activation,  A_mart, b_mart, stats_at_t_mart)

    np.random.seed(basis_seed + 1)

    A_reg = ( std_basis_reg * np.random.randn(nneurons_reg - 1, nb_stocks) ).astype(np.float32)
    b_reg = ( std_basis_reg * np.random.randn(nneurons_reg - 1) ).astype(np.float32)
    value = (payoff(policy_set[-1], strike_price)* np.exp(-interest_rate * maturity)- martingale_regression[-1])  # Modified payoff 
    reg_coefficients = np.zeros(( exercise_dates - 1, nneurons_reg)) 
    # We are not modified the ITM paths since we just modified the continuation value bc of (7)
    stats_at_t_reg = [None] * (exercise_dates - 1)
    for t in range(exercise_dates - 1, 0, -1):
        # " Better to use (7) instead of (4)"
        payoff_at_t = payoff(policy_set[t*subintervals, :], strike_price) * np.exp( -interest_rate * interval_exercise * t)
        basis_at_t, stats_at_t_reg[t-1] = get_basis_random(policy_set[t*subintervals, :], activation, A_reg, b_reg)
        itm = payoff_at_t > 0
        target = value + martingale_regression[t-1,:]
        Y_itm = target[itm]
        X_itm_vander = basis_at_t[:, itm]

        reg_coefficients[t-1] = fit_regression_coefficients( X_itm_vander.T, Y_itm, ridge_alpha)
        continuation = np.einsum('ij,i->j', basis_at_t, reg_coefficients[t-1, :])
        exercise = (payoff_at_t > continuation) & itm
        value = np.where(exercise, payoff_at_t - martingale_regression[t-1, :], value)
    
    # Pricing phase
    npaths_pricing = price_set.shape[2]
    martingale_pricing = determine_martingale(price_set, fine_dt, interest_rate, dividend, exercise_dates, subintervals, coefficients_martingale, activation, A_mart, b_mart, stats_at_t_mart)

    matrix_exercise_time = np.zeros((exercise_dates, npaths_pricing), dtype=bool)
    has_exercised = np.zeros(npaths_pricing, dtype=bool)

    for t in range(0, exercise_dates-1):
        payoff_at_t = payoff(price_set[(t+1)*subintervals, :], strike_price) * np.exp( -interest_rate * interval_exercise * (t + 1))
        itm = payoff_at_t > 0
        basis_at_t, _ = get_basis_random(price_set[(t+1)*subintervals, :], activation, A_reg, b_reg, stats_at_t_reg[t])
        continuation = np.einsum('ij,i->j', basis_at_t, reg_coefficients[t, :])
        exercise_condition = (payoff_at_t > continuation) & (~has_exercised) & itm
        matrix_exercise_time[t, exercise_condition] = True
        has_exercised |= exercise_condition

    matrix_exercise_time[-1, ~has_exercised] = True

    intrinsic_value_pricing = np.zeros((exercise_dates + 1, npaths_pricing))
    for k in range(exercise_dates + 1):
        q = k * subintervals
        intrinsic_value_pricing[k] = (payoff(price_set[q], strike_price)* np.exp(-interest_rate * interval_exercise * k))
    payoff_control_variate = np.sum(intrinsic_value_pricing[1:] * matrix_exercise_time,axis=0) # Payoff of the control variate using the optimal stopping rule
    martingale_control_variate = np.sum(martingale_pricing * matrix_exercise_time,axis=0) # Martingale control variate using the optimal stopping rule
    denominator = np.sum(martingale_control_variate**2)
    if denominator > 0:
        coeff_lambda = np.sum(payoff_control_variate * martingale_control_variate) / denominator
    else:
        coeff_lambda = 0.0

    # Determining lower bound
    controlled_payoff = payoff_control_variate - coeff_lambda * martingale_control_variate
    lower_bound = np.mean(controlled_payoff)
    lower_uncertainty = np.std(controlled_payoff, ddof=1) / np.sqrt(npaths_pricing)

    immediate_exercise = payoff(price_set[0, :], strike_price)[0]

    if immediate_exercise >= lower_bound:
        lower_bound = immediate_exercise
        lower_uncertainty = 0.0

    # Diagnostics
    # Compute the raw price and standard error without control variate
    raw_price = np.mean(payoff_control_variate)
    raw_standard_error = np.std(payoff_control_variate, ddof=1) / np.sqrt(npaths_pricing)

    variance_Y = np.var( payoff_control_variate, ddof=1)
    variance_controlled = np.var( controlled_payoff, ddof=1)

    if variance_Y > 0:
        variance_reduction = variance_controlled / variance_Y 
    else:
        variance_reduction = np.inf

    if np.std(payoff_control_variate, ddof=1) > 0 and np.std(martingale_control_variate, ddof=1) > 0:
        correlation = np.corrcoef(payoff_control_variate, martingale_control_variate)[0, 1]
    else:
        correlation = np.nan
    diagnostics = { "raw_price": raw_price, "raw_standard_error": raw_standard_error,
                   "variance_reduction": variance_reduction, "correlation_payoff_martingale": correlation, 
                   "lambda_cv": coeff_lambda, "mean_martingale": np.mean(martingale_control_variate)}
    
    
    # Determining the upper bound
    max_per_path = np.max(intrinsic_value_pricing - np.vstack([np.zeros((1, npaths_pricing)), martingale_pricing]), axis=0)
    upper_bound = np.mean(max_per_path)
    upper_uncertainty = np.std(max_per_path, ddof=1) / np.sqrt(npaths_pricing)  

    return lower_bound, lower_uncertainty, upper_bound, upper_uncertainty, diagnostics
    