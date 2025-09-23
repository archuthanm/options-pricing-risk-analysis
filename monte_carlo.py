import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from black_scholes import bs_delta as delta, bs_gamma as gamma


#---------ANALYSIS---------------

def compute_greeks(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Computes the Monte Carlo estimates and standard errors for option Greeks: Delta, Gamma, Vega, Theta, and Rho.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (
            MCdelta (float), SEdelta (float),
            MCgamma (float), SEgamma (float),
            MCvega (float), SEvega (float),
            MCtheta (float), SEtheta (float),
            MCrho (float), SErho (float)
        )
    """
    MCdelta, SEdelta = mc_delta(S, X, vol, r, N, M, Z, T, type)
    MCgamma, SEgamma = mc_gamma(S, X, vol, r, N, M, Z, T, type)
    MCvega, SEvega = mc_vega(S, X, vol, r, N, M, Z, T, type)
    MCtheta, SEtheta = mc_theta(S, X, vol, r, N, M, Z, T, type)
    MCrho, SErho = mc_rho(S, X, vol, r, N, M, Z, T, type)
    return MCdelta, SEdelta, MCgamma, SEgamma, MCvega, SEvega, MCtheta, SEtheta, MCrho, SErho

def benchmark_mc_variants(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> pd.DataFrame:
    """
    Benchmarks various Monte Carlo option pricing methods and their variance reduction techniques.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        pd.DataFrame: DataFrame summarizing standard error and computation time for each Monte Carlo variant.
    """
    C02, SE2, comp2 = mc_baseline(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    C03, SE3, comp3 = mc_antithetic(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    C04, SE4, comp4 = mc_delta_control(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    C05, SE5, comp5 = mc_gamma_control(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    C06, SE6, comp6 = mc_antithetic_delta(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    C07, SE7, comp7 = mc_antithetic_delta_gamma(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    results = [
        {
            "Function": "Vectorized Baseline",
            "Standard Error": SE2,
            "Computation Time": comp2,
            "Standard Error Reduction Multiple": SE2/SE2,
            "Relative Computation Time": comp2/comp2,
        },
        {
            "Function": "Antithetic Variates",
            "Standard Error": SE3,
            "Computation Time": comp3,
            "Standard Error Reduction Multiple": SE2/SE3,
            "Relative Computation Time": comp3/comp2,
        },
        {
            "Function": "Delta-based Control Variates",
            "Standard Error": SE4,
            "Computation Time": comp4,
            "Standard Error Reduction Multiple": SE2/SE4,
            "Relative Computation Time": comp4/comp2,
        },
        {
            "Function": "Gamma-based Control Variates",
            "Standard Error": SE5,
            "Computation Time": comp5,
            "Standard Error Reduction Multiple": SE2/SE5,
            "Relative Computation Time": comp5/comp2,
        },
        {
            "Function": "Antithetic AND Delta Variates",
            "Standard Error": SE6,
            "Computation Time": comp6,
            "Standard Error Reduction Multiple": SE2/SE6,
            "Relative Computation Time": comp6/comp2,
        },
        {
            "Function": "Antithetic, Delta AND Gamma Variates",
            "Standard Error": SE7,
            "Computation Time": comp7,
            "Standard Error Reduction Multiple": SE2/SE7,
            "Relative Computation Time": comp7/comp2,
        }
    ]
    pd.set_option('display.max_colwidth', None)
    columns = [
        "Function",
        "Standard Error",
        "Computation Time",
        "Standard Error Reduction Multiple",
        "Relative Computation Time",
    ]
    df = pd.DataFrame(results, columns=columns)
    return df


#---------MONTE CARLO SIMULATIONS----------

def mc_baseline(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Baseline vectorized Monte Carlo implementation for European option pricing.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Baseline vectorized Monte Carlo implementation (efficient European option pricing).
    # Precompute constants
    dt = T/N
    # Time step length (years)
    drift_dt = (r - vol**2/2)*dt
    # Drift component (risk-neutral measure)
    volsdt = vol*np.sqrt(dt)
    # Diffusion component
    lnS = np.log(S)
    # Initial log-price
    # Log-price increments
    delta_lnst = drift_dt + volsdt*Z
    # Cumulative log-price paths
    lnSt = lnS + np.cumsum(delta_lnst, axis=0)
    # Include initial log-price as row 0
    lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), lnSt))
    # Convert log-prices to asset prices
    ST = np.exp(lnSt)
    # Option payoff at maturity
    if type == "C":
        discounted_payoff = np.exp(-r*T) * np.maximum(0, ST[-1] - X)
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * np.maximum(0, X - ST[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    # Monte Carlo price (mean discounted payoff)
    C0 = np.sum(discounted_payoff) / M
    # Standard deviation of discounted payoffs
    sigma = np.sqrt(np.sum((discounted_payoff - C0)**2) / (M - 1))
    # Standard error of estimate
    SE = sigma/np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time

def mc_antithetic(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Monte Carlo option pricing using antithetic variates for variance reduction.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Precompute constants
    dt = T/N
    drift_dt = (r - vol**2/2)*dt
    volsdt = vol*np.sqrt(dt)
    lnS = np.log(S)
    # Log-price increments (positive and negative shocks)
    delta_lnSt1 = drift_dt + volsdt*Z
    delta_lnSt2 = drift_dt - volsdt*Z
    lnSt1 = lnS + np.cumsum(delta_lnSt1, axis=0)
    lnSt2 = lnS + np.cumsum(delta_lnSt2, axis=0)
    # Asset price paths under positive/negative shocks
    ST1 = np.exp(lnSt1)
    ST2 = np.exp(lnSt2)
    # Option payoff at maturity
    if type == "C":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, ST1[-1] - X) + np.maximum(0, ST2[-1] - X))/2
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, X - ST1[-1]) + np.maximum(0, X - ST2[-1]))/2
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    C0 = np.sum(discounted_payoff) / M
    sigma = np.sqrt(np.sum((discounted_payoff - C0)**2) / (M - 1))
    SE = sigma/np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time

def mc_delta_control(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Monte Carlo option pricing using Delta-based control variates for variance reduction.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Precompute constants
    dt = T / N
    # Time step length (years)
    drift_dt = (r - vol**2 / 2) * dt
    # Drift component (risk-neutral measure)
    volsdt = vol * np.sqrt(dt)
    # Diffusion component
    erdt = np.exp(r * dt)
    # Forward factor
    beta1 = -1  # Control variate coefficient

    # Log-price increments
    delta_St = drift_dt + volsdt * Z
    # Cumulative asset price paths
    ST = S * np.cumprod(np.exp(delta_St), axis=0)
    ST = np.concatenate((np.full(shape=(1, M), fill_value=S), ST))
    # Delta at each time step (excluding t=0)
    deltaST = delta(r, ST[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    # Cumulative Delta-based control variate adjustments
    cv = np.cumsum(deltaST * (ST[1:] - ST[:-1] * erdt), axis=0)
    # Option payoff adjusted with Delta control variate
    if type == "C":
        discounted_payoff = np.exp(-r * T) * (np.maximum(0, ST[-1] - X) + beta1 * cv[-1])
    elif type == "P":
        discounted_payoff = np.exp(-r * T) * (np.maximum(0, X - ST[-1]) + beta1 * cv[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    # Monte Carlo price (mean discounted payoff)
    C0 = np.sum(discounted_payoff) / M
    # Standard deviation of discounted payoffs
    sigma = np.sqrt(np.sum((discounted_payoff - C0) ** 2) / (M - 1))
    # Standard error of estimate
    SE = sigma / np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time

def mc_gamma_control(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Monte Carlo option pricing using Gamma-based control variates for variance reduction.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Precompute constants
    dt = T / N
    # Time step length (years)
    drift_dt = (r - vol**2 / 2) * dt
    # Drift component (risk-neutral measure)
    volsdt = vol * np.sqrt(dt)
    # Diffusion component
    erdt = np.exp(r * dt)
    # Forward factor
    ergamma = np.exp((2 * r + vol**2) * dt) - 2 * erdt + 1
    beta2 = -0.5  # Control variate coefficient

    # Log-price increments
    delta_St = drift_dt + volsdt * Z
    # Cumulative asset price paths
    ST = S * np.cumprod(np.exp(delta_St), axis=0)
    ST = np.concatenate((np.full(shape=(1, M), fill_value=S), ST))
    # Gamma at each time step
    gammaST = gamma(r, ST[:-1].T, X, np.linspace(T, dt, N), vol).T
    # Cumulative Gamma-based control variate adjustments
    cv2 = np.cumsum(gammaST * ((ST[1:] - ST[:-1]) ** 2 - ergamma * ST[:-1] ** 2), axis=0)
    # Option payoff adjusted with Gamma control variate
    if type == "C":
        discounted_payoff = np.exp(-r * T) * (np.maximum(0, ST[-1] - X) + beta2 * cv2[-1])
    elif type == "P":
        discounted_payoff = np.exp(-r * T) * (np.maximum(0, X - ST[-1]) + beta2 * cv2[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    # Monte Carlo price (mean discounted payoff)
    C0 = np.sum(discounted_payoff) / M
    # Standard deviation of discounted payoffs
    sigma = np.sqrt(np.sum((discounted_payoff - C0) ** 2) / (M - 1))
    # Standard error of estimate
    SE = sigma / np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time

def mc_antithetic_delta(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Monte Carlo option pricing using antithetic variates and Delta-based control variates.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Precompute constants
    dt = T / N
    drift_dt = (r - vol**2 / 2) * dt
    volsdt = vol * np.sqrt(dt)
    erdt = np.exp(r * dt)
    beta1 = -1  # Control variate coefficient

    # Log-price increments for antithetic pairs
    delta_St1 = drift_dt + volsdt * Z
    delta_St2 = drift_dt - volsdt * Z
    # Cumulative asset price paths for each antithetic pair
    ST1 = S * np.cumprod(np.exp(delta_St1), axis=0)
    ST2 = S * np.cumprod(np.exp(delta_St2), axis=0)
    ST1 = np.concatenate((np.full(shape=(1, M), fill_value=S), ST1))
    ST2 = np.concatenate((np.full(shape=(1, M), fill_value=S), ST2))
    # Delta at each time step for both paths
    deltaST1 = delta(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    deltaST2 = delta(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    # Cumulative Delta-based control variate adjustments
    cv1 = np.cumsum(deltaST1 * (ST1[1:] - ST1[:-1] * erdt), axis=0)
    cv2 = np.cumsum(deltaST2 * (ST2[1:] - ST2[:-1] * erdt), axis=0)
    # Option payoff adjusted with Delta control variate, averaged over antithetic pairs
    if type == "C":
        discounted_payoff = 0.5 * np.exp(-r * T) * (
            np.maximum(0, ST1[-1] - X) + beta1 * cv1[-1] +
            np.maximum(0, ST2[-1] - X) + beta1 * cv2[-1]
        )
    elif type == "P":
        discounted_payoff = 0.5 * np.exp(-r * T) * (
            np.maximum(0, X - ST1[-1]) + beta1 * cv1[-1] +
            np.maximum(0, X - ST2[-1]) + beta1 * cv2[-1]
        )
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    # Monte Carlo price (mean discounted payoff)
    C0 = np.sum(discounted_payoff) / M
    # Standard deviation of discounted payoffs
    sigma = np.sqrt(np.sum((discounted_payoff - C0) ** 2) / (M - 1))
    # Standard error of estimate
    SE = sigma / np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time

def mc_antithetic_delta_gamma(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, return_payoffs: bool = False
) -> tuple[float, float, float] | np.ndarray:
    """
    Monte Carlo option pricing using antithetic variates, Delta-based, and Gamma-based control variates.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (option price (float), standard error (float), computation time (float, seconds))
    """
    start_time = time.time()
    # Precompute constants
    dt = T / N
    drift_dt = (r - vol**2 / 2) * dt
    volsdt = vol * np.sqrt(dt)
    erdt = np.exp(r * dt)
    ergamma = np.exp((2 * r + vol**2) * dt) - 2 * erdt + 1
    beta1 = -1    # Delta control variate coefficient
    beta2 = -0.5  # Gamma control variate coefficient

    # Log-price increments for antithetic pairs
    delta_St1 = drift_dt + volsdt * Z
    delta_St2 = drift_dt - volsdt * Z
    # Cumulative asset price paths
    ST1 = S * np.cumprod(np.exp(delta_St1), axis=0)
    ST2 = S * np.cumprod(np.exp(delta_St2), axis=0)
    ST1 = np.concatenate((np.full(shape=(1, M), fill_value=S), ST1))
    ST2 = np.concatenate((np.full(shape=(1, M), fill_value=S), ST2))
    # Delta and Gamma at each time step for both paths
    deltaST1 = delta(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    deltaST2 = delta(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    gammaST1 = gamma(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol).T
    gammaST2 = gamma(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol).T
    # Cumulative Delta and Gamma-based control variate adjustments
    cv1d = np.cumsum(deltaST1 * (ST1[1:] - ST1[:-1] * erdt), axis=0)
    cv2d = np.cumsum(deltaST2 * (ST2[1:] - ST2[:-1] * erdt), axis=0)
    cv1g = np.cumsum(gammaST1 * ((ST1[1:] - ST1[:-1]) ** 2 - ergamma * ST1[:-1] ** 2), axis=0)
    cv2g = np.cumsum(gammaST2 * ((ST2[1:] - ST2[:-1]) ** 2 - ergamma * ST2[:-1] ** 2), axis=0)
    # Option payoff adjusted with Delta and Gamma control variates, averaged over antithetic pairs
    if type == "C":
        discounted_payoff = 0.5 * np.exp(-r * T) * (
            np.maximum(0, ST1[-1] - X) + beta1 * cv1d[-1] + beta2 * cv1g[-1] +
            np.maximum(0, ST2[-1] - X) + beta1 * cv2d[-1] + beta2 * cv2g[-1]
        )
    elif type == "P":
        discounted_payoff = 0.5 * np.exp(-r * T) * (
            np.maximum(0, X - ST1[-1]) + beta1 * cv1d[-1] + beta2 * cv1g[-1] +
            np.maximum(0, X - ST2[-1]) + beta1 * cv2d[-1] + beta2 * cv2g[-1]
        )
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    # Estimate option value and standard error
    # Monte Carlo price (mean discounted payoff)
    C0 = np.sum(discounted_payoff) / M
    # Standard deviation of discounted payoffs
    sigma = np.sqrt(np.sum((discounted_payoff - C0) ** 2) / (M - 1))
    # Standard error of estimate
    SE = sigma / np.sqrt(M)
    computation_time = time.time() - start_time
    if return_payoffs:
        return discounted_payoff
    return C0, SE, computation_time


#---------MONTE CARLO GREEKS----------

def mc_delta(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:
    """
    Estimates the Delta of a European option using a pathwise Monte Carlo method with antithetic variates.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.

    Returns:
        tuple: (Delta estimate (float), standard error (float))
    """
    # Precompute constants
    dt = T / N
    # Time step length (years)
    drift_dt = (r - vol**2 / 2) * dt
    # Drift component (risk-neutral measure)
    volsdt = vol * np.sqrt(dt)
    # Diffusion component
    erdt = np.exp(r * dt)
    ergamma = np.exp((2 * r + vol**2) * dt) - 2 * erdt + 1
    # Antithetic log-price increments
    delta_St1 = drift_dt + volsdt * Z
    delta_St2 = drift_dt - volsdt * Z
    # Asset price paths
    ST1 = S * np.cumprod(np.exp(delta_St1), axis=0)
    ST2 = S * np.cumprod(np.exp(delta_St2), axis=0)
    ST1 = np.concatenate((np.full((1, M), S), ST1))
    ST2 = np.concatenate((np.full((1, M), S), ST2))
    # Pathwise Delta estimate at t=0
    if type == "C":
        delta_paths = 0.5 * ((ST1[-1] > X).astype(float) * ST1[-1] / S +
                             (ST2[-1] > X).astype(float) * ST2[-1] / S)
    elif type == "P":
        delta_paths = 0.5 * (-(ST1[-1] < X).astype(float) * ST1[-1] / S -
                             (ST2[-1] < X).astype(float) * ST2[-1] / S)
    else:
        raise ValueError("type must be 'C' or 'P'")
    # Discounted back to present
    delta_paths *= np.exp(-r * T)
    delta_exp = np.mean(delta_paths)
    SE = np.std(delta_paths, ddof=1) / np.sqrt(M)
    return float(delta_exp), float(SE)

def mc_gamma(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.5) -> tuple[float, float]:
    """
    Estimates the Gamma of a European option using a central finite difference of Monte Carlo prices.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.
        h (float, optional): Spot price bump for finite difference. Default is 0.5.

    Returns:
        tuple: (Gamma estimate (float), standard error (float))
    """
    # Price with upward spot bump
    C1, SE1, comp1 = mc_antithetic_delta_gamma(S + h, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    # Price at baseline spot
    C2, SE2, comp2 = mc_antithetic_delta_gamma(S, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    # Price with downward spot bump
    C3, SE3, comp3 = mc_antithetic_delta_gamma(S - h, X, vol, r, N, M, Z, T, type, return_payoffs=False)
    # Central difference approximation
    gamma_exp = (C1 - 2 * C2 + C3) / (h ** 2)
    # Standard error propagation
    SE = np.sqrt(SE1 ** 2 + 4 * SE2 ** 2 + SE3 ** 2) / (h ** 2)
    return float(gamma_exp), float(SE)

def mc_vega(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.01) -> tuple[float, float]:
    """
    Estimates the Vega of a European option using a central finite difference of Monte Carlo prices.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.
        h (float, optional): Volatility bump for finite difference. Default is 0.01.

    Returns:
        tuple: (Vega estimate per 1% vol (float), standard error (float))
    """
    # Price with upward volatility bump
    C1, SE1, comp1 = mc_antithetic_delta_gamma(S, X, vol + h, r, N, M, Z, T, type, return_payoffs=False)
    # Price with downward volatility bump
    C2, SE2, comp2 = mc_antithetic_delta_gamma(S, X, vol - h, r, N, M, Z, T, type, return_payoffs=False)
    # Central difference approximation
    vega_exp = (C1 - C2) / (2 * h)
    # Standard error propagation
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / (2 * h)
    return float(vega_exp / 100), float(SE / 100)

def mc_theta(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 1 / 365) -> tuple[float, float]:
    """
    Estimates the Theta of a European option using a central finite difference of Monte Carlo prices.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.
        h (float, optional): Time bump for finite difference (years). Default is 1/365 (1 day).

    Returns:
        tuple: (Theta estimate per day (float), standard error (float))
    """
    # Price with extended maturity
    C2, SE2, comp1 = mc_antithetic_delta_gamma(S, X, vol, r, N, M, Z, T + h, type, return_payoffs=False)
    # Price with reduced maturity
    if T - h <= 0:
        raise ValueError("T - h must be positive for finite difference Theta calculation.")
    C1, SE1, comp2 = mc_antithetic_delta_gamma(S, X, vol, r, N, M, Z, T - h, type, return_payoffs=False)
    # Central difference approximation
    theta_exp = (C1 - C2) / (2 * h)
    # Standard error propagation
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / (2 * h)
    return float(theta_exp / 365), float(SE / 365)

def mc_rho(
    S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.01 ) -> tuple[float, float]:
    """
    Estimates the Rho of a European option using a central finite difference of Monte Carlo prices.

    Args:
        S (float): Underlying asset price.
        X (float): Option strike price.
        vol (float): Annualized volatility.
        r (float): Risk-free interest rate.
        N (int): Number of time discretization steps.
        M (int): Number of Monte Carlo paths.
        Z (np.ndarray): Standard normal draws of shape (N, M).
        T (float): Time to maturity (years).
        type (str): Option type, "C" for Call or "P" for Put.
        h (float, optional): Interest rate bump for finite difference. Default is 0.01.

    Returns:
        tuple: (Rho estimate per 1% rate (float), standard error (float))
    """
    # Price with upward rate bump
    C1, SE1, comp1 = mc_antithetic_delta_gamma(S, X, vol, r + h, N, M, Z, T, type, return_payoffs=False)
    # Price with downward rate bump
    C2, SE2, comp2 = mc_antithetic_delta_gamma(S, X, vol, r - h, N, M, Z, T, type, return_payoffs=False)
    # Central difference approximation
    rho_exp = (C1 - C2) / (2 * h)
    # Standard error propagation
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / (2 * h)
    return float(rho_exp / 100), float(SE / 100)