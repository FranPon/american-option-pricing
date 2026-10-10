from math import comb
import numpy as np
from scipy.stats import norm
from basis import poly_basis_ram
from standardization_ridge import fit_regression_coefficients


def indicator_cell_indices(assets, P, stats=None):

    assets = np.asarray(assets)
    nb_stocks, paths = assets.shape

    log_asset = np.log(assets)
    if stats is None:
        mu = np.mean(log_asset, axis=1, keepdims=True)
        std = np.std(log_asset, axis=1, keepdims=True)
        stats = (mu, std)
    else:
        mu, std = stats

    std_safe = np.maximum(std, 1e-8)
    probabilities = norm.cdf((log_asset - mu) / std_safe)
    probabilities = np.minimum(probabilities, np.nextafter(1.0, 0.0))

    bins = np.floor(probabilities * P).astype(np.int64)
    np.clip(bins, 0, P - 1, out=bins)

    cells = bins[0].copy()
    for stock in range(1, nb_stocks):
        cells *= P
        cells += bins[stock]

    if cells.shape != (paths,):
        raise RuntimeError("Unexpected partition-index shape.")

    return cells, stats


def fit_local_martingale_coefficients( cells, increments, target, number_of_cells, ridge_alpha):
    cells = np.asarray(cells, dtype=np.int64)
    increments = np.asarray(increments, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    nb_stocks, paths = increments.shape
    if cells.shape != (paths,):
        raise ValueError("cells and increments contain different path counts.")
    if target.shape != (paths,):
        raise ValueError("target and increments contain different path counts.")

    counts = np.bincount(cells, minlength=number_of_cells)
    occupied = counts > 0

    gram = np.zeros( (number_of_cells, nb_stocks, nb_stocks), dtype=np.float64) 
    rhs = np.zeros((number_of_cells, nb_stocks), dtype=np.float64)

    for k in range(nb_stocks):
        rhs[:, k] = np.bincount( cells, weights=increments[k] * target, minlength=number_of_cells)
        for ell in range(k, nb_stocks):
            cross_product = np.bincount(cells, weights=increments[k] * increments[ell], minlength=number_of_cells)
            gram[:, k, ell] = cross_product
            gram[:, ell, k] = cross_product

    coefficients = np.zeros( (number_of_cells, nb_stocks), dtype=np.float64)

    occupied_gram = gram[occupied]
    occupied_rhs = rhs[occupied]

    if ridge_alpha != 0.0:
        diagonal = np.arange(nb_stocks)
        occupied_gram[:, diagonal, diagonal] += ridge_alpha

    try:
        coefficients[occupied] = np.linalg.solve(occupied_gram, occupied_rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        occupied_cells = np.flatnonzero(occupied)
        for local_index, cell in enumerate(occupied_cells):
            coefficients[cell] = np.linalg.lstsq( occupied_gram[local_index], occupied_rhs[local_index], rcond=None)[0]

    return coefficients


def determine_martingale_standard_local( assets, fine_dt, interest_rate, dividend, P, exercise_dates, subintervals, coefficients, stats):

    nb_stocks, paths = assets.shape[1], assets.shape[2]
    number_of_cells = P**nb_stocks
    martingale_increments = np.zeros((exercise_dates, paths))

    for t in range(exercise_dates - 1, -1, -1):
        increment_t = np.zeros(paths)

        for j in range(subintervals):
            q = t * subintervals + j
            cells, _ = indicator_cell_indices(assets[q], P, stats[q])

            G = (assets[q + 1] * np.exp((dividend - interest_rate) * fine_dt * (q + 1)) 
                - assets[q] * np.exp((dividend - interest_rate) * fine_dt * q))

            alpha = coefficients[t, j].reshape(number_of_cells, nb_stocks)
            increment_t += np.einsum( "pd,dp->p", alpha[cells], G, optimize=True)

        martingale_increments[t] = increment_t

    return np.cumsum(martingale_increments, axis=0)


def computing_alfonsi_martingale_local( exercise_dates, subintervals, P, interest_rate, dividend, 
        interval_exercise, fine_dt, strike_price, payoff, asset_exercise, assets_fine, ridge_alpha):

    nb_stocks, npaths_training = assets_fine.shape[1], assets_fine.shape[2]
    number_of_cells = P**nb_stocks
    number_of_coefficients = number_of_cells * nb_stocks

    coefficients = np.zeros( (exercise_dates, subintervals, number_of_coefficients), dtype=np.float64)
    martingale_increments = np.zeros((exercise_dates, npaths_training))
    stats_at_t = [None] * (exercise_dates * subintervals)

    for t in range(exercise_dates - 1, -1, -1):
        Z = np.stack(
            [payoff(asset_exercise[k], strike_price) * np.exp(-interest_rate * interval_exercise * k) 
             for k in range(t + 1, exercise_dates + 1)],axis=0)

        if t + 1 < exercise_dates:
            sum_to_remove = np.vstack([np.zeros((1, npaths_training)),np.cumsum(martingale_increments[t + 1 :], axis=0)])
        else:
            sum_to_remove = np.zeros((1, npaths_training))

        target = np.max(Z - sum_to_remove, axis=0)
        increment_t = np.zeros(npaths_training)

        for j in range(subintervals):
            q = t * subintervals + j
            cells, stats_at_t[q] = indicator_cell_indices( assets_fine[q], P)

            G = ( assets_fine[q + 1] * np.exp((dividend - interest_rate) * fine_dt * (q + 1))
                - assets_fine[q] * np.exp((dividend - interest_rate) * fine_dt * q))

            if t == 0 and j == 0:
                beta = np.linalg.lstsq(G.T, target, rcond=None)[0]
                alpha = np.zeros((number_of_cells, nb_stocks))
                alpha[np.unique(cells)] = beta
            else:
                alpha = fit_local_martingale_coefficients( cells, G, target, number_of_cells, ridge_alpha)

            coefficients[t, j] = alpha.reshape(-1)
            increment_t += np.einsum( "pd,dp->p", alpha[cells], G, optimize=True)

        martingale_increments[t] = increment_t

    martingale = np.cumsum(martingale_increments, axis=0)
    return coefficients, martingale, stats_at_t


def american_AL_PRIMAL_DUAL_standard_local( payoff, strike_price, interest_rate, dividend, maturity,
    exercise_dates, subintervals, P, max_deg, martingale_set, policy_set, price_set, ridge_alpha):

    fine_dt = maturity / (exercise_dates * subintervals)
    interval_exercise = maturity / exercise_dates
    nb_stocks = martingale_set.shape[1]

    martingale_set_exercise = martingale_set[::subintervals]
    coefficients_martingale, _, stats_at_t_mart = computing_alfonsi_martingale_local( exercise_dates, subintervals, P,
        interest_rate, dividend, interval_exercise, fine_dt, strike_price, payoff, martingale_set_exercise,
        martingale_set, ridge_alpha)

    martingale_regression = determine_martingale_standard_local( policy_set, fine_dt, interest_rate, dividend, P,
        exercise_dates, subintervals, coefficients_martingale, stats_at_t_mart)

    nb_basis = comb(nb_stocks + max_deg, max_deg)
    value = (
        payoff(policy_set[-1], strike_price) * np.exp(-interest_rate * maturity)
        - martingale_regression[-1]
    )
    reg_coefficients = np.zeros((exercise_dates - 1, nb_basis))

    for t in range(exercise_dates - 1, 0, -1):
        payoff_at_t = (
            payoff(policy_set[t * subintervals], strike_price)
            * np.exp(-interest_rate * interval_exercise * t)
        )
        basis_at_t = poly_basis_ram(policy_set[t * subintervals], max_deg)
        itm = payoff_at_t > 0
        target = value + martingale_regression[t - 1]
        Y_itm = target[itm]
        X_itm_vander = basis_at_t[:, itm]

        reg_coefficients[t - 1] = fit_regression_coefficients( X_itm_vander.T, Y_itm, ridge_alpha)
        continuation = np.einsum( "ij,i->j", basis_at_t, reg_coefficients[t - 1])
        exercise = (payoff_at_t > continuation) & itm
        value = np.where( exercise, payoff_at_t - martingale_regression[t - 1], value)

    npaths_pricing = price_set.shape[2]
    martingale_pricing = determine_martingale_standard_local( price_set, fine_dt, interest_rate, dividend,
        P, exercise_dates, subintervals, coefficients_martingale, stats_at_t_mart)

    matrix_exercise_time = np.zeros( (exercise_dates, npaths_pricing), dtype=bool)
    has_exercised = np.zeros(npaths_pricing, dtype=bool)

    for t in range(exercise_dates - 1):
        payoff_at_t = ( payoff(price_set[(t + 1) * subintervals], strike_price) * np.exp(-interest_rate * interval_exercise * (t + 1)))
        itm = payoff_at_t > 0
        basis_at_t = poly_basis_ram( price_set[(t + 1) * subintervals], max_deg)
        continuation = np.einsum( "ij,i->j", basis_at_t, reg_coefficients[t])
        exercise_condition = ( (payoff_at_t > continuation) & (~has_exercised) & itm)
        matrix_exercise_time[t, exercise_condition] = True
        has_exercised |= exercise_condition

    matrix_exercise_time[-1, ~has_exercised] = True

    intrinsic_value_pricing = np.zeros((exercise_dates + 1, npaths_pricing))
    for k in range(exercise_dates + 1):
        q = k * subintervals
        intrinsic_value_pricing[k] = ( payoff(price_set[q], strike_price) * np.exp(-interest_rate * interval_exercise * k))

    payoff_control_variate = np.sum( intrinsic_value_pricing[1:] * matrix_exercise_time, axis=0)
    martingale_control_variate = np.sum( martingale_pricing * matrix_exercise_time, axis=0)

    denominator = np.sum(martingale_control_variate**2)
    if denominator > 0:
        coeff_lambda = np.sum(payoff_control_variate * martingale_control_variate) / denominator
    else:
        coeff_lambda = 0.0

    controlled_payoff = payoff_control_variate - coeff_lambda * martingale_control_variate

    lower_bound = np.mean(controlled_payoff)
    lower_uncertainty = np.std(controlled_payoff, ddof=1) / np.sqrt(npaths_pricing)


    immediate_exercise = payoff(price_set[0], strike_price)[0]
    if immediate_exercise >= lower_bound:
        lower_bound = immediate_exercise
        lower_uncertainty = 0.0

    raw_price = np.mean(payoff_control_variate)
    raw_standard_error = np.std(payoff_control_variate, ddof=1) / np.sqrt(npaths_pricing)


    variance_Y = np.var(payoff_control_variate, ddof=1)
    variance_controlled = np.var(controlled_payoff, ddof=1)
    variance_reduction = ( variance_controlled / variance_Y if variance_Y > 0 else np.inf )

    payoff_std = np.std(payoff_control_variate, ddof=1)
    martingale_std = np.std(martingale_control_variate, ddof=1)
    if payoff_std > 0 and martingale_std > 0:
        correlation = np.corrcoef( payoff_control_variate, martingale_control_variate )[0, 1]
    else:
        correlation = np.nan

    diagnostics = {
        "raw_price": raw_price,
        "raw_standard_error": raw_standard_error,
        "variance_reduction": variance_reduction,
        "correlation_payoff_martingale": correlation,
        "lambda_cv": coeff_lambda,
        "mean_martingale": np.mean(martingale_control_variate),
    }

    max_per_path = np.max(intrinsic_value_pricing - np.vstack([np.zeros((1, npaths_pricing)),martingale_pricing]),axis=0)
    upper_bound = np.mean(max_per_path)
    upper_uncertainty = np.std(max_per_path, ddof=1) / np.sqrt(npaths_pricing)


    return lower_bound, lower_uncertainty, upper_bound, upper_uncertainty, diagnostics