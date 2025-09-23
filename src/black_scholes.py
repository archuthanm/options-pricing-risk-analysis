"""
Module implementing the Black-Scholes formula and its associated Greeks for European options pricing.

Provides functions to calculate option price and the Greeks: Delta, Gamma, Vega, Theta, and Rho.
"""

import numpy as np
from scipy.stats import norm


def blackScholes(r: float, S: float, X: float, T: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Black-Scholes price of a European call or put option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'C' for Call option, 'P' for Put option.

    Returns:
        float: Theoretical option price.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    return price


def bs_delta(r: float, S: float, X: float, T: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Delta Greek of a European option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'C' for Call option, 'P' for Put option.

    Returns:
        float: Delta of the option.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "C":
        delta = norm.cdf(d1)
    elif option_type == "P":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    return delta


def bs_gamma(r: float, S: float, X: float, T: float, sigma: float) -> float:
    """
    Calculate the Gamma Greek of a European option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.

    Returns:
        float: Gamma of the option.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def bs_vega(r: float, S: float, X: float, T: float, sigma: float) -> float:
    """
    Calculate the Vega Greek of a European option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.

    Returns:
        float: Vega of the option, representing sensitivity to a 1% change in implied volatility.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega / 100  # Scaled to 1% change in volatility


def bs_theta(r: float, S: float, X: float, T: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Theta Greek of a European option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'C' for Call option, 'P' for Put option.

    Returns:
        float: Theta of the option, representing time decay per day.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * X * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "P":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * X * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")

    return theta / 365  # Per day decay


def bs_rho(r: float, S: float, X: float, T: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Rho Greek of a European option.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'C' for Call option, 'P' for Put option.

    Returns:
        float: Rho of the option, representing sensitivity to a 1% change in interest rates.
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        rho = X * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        rho = -X * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")

    return rho / 100  # Scaled to 1% change in interest rates


def compute_greeks(r: float, S: float, X: float, T: float, sigma: float, option_type: str) -> tuple[float, float, float, float, float]:
    """
    Compute all primary Greeks for a European option using the Black-Scholes model.

    Parameters:
        r (float): Risk-free interest rate.
        S (float): Current underlying stock price.
        X (float): Strike price.
        T (float): Time to expiration in years.
        sigma (float): Volatility of the underlying asset.
        option_type (str): 'C' for Call option, 'P' for Put option.

    Returns:
        tuple: (Delta, Gamma, Vega, Theta, Rho)
    """
    delta = bs_delta(r, S, X, T, sigma, option_type)
    gamma = bs_gamma(r, S, X, T, sigma)
    vega = bs_vega(r, S, X, T, sigma)
    theta = bs_theta(r, S, X, T, sigma, option_type)
    rho = bs_rho(r, S, X, T, sigma, option_type)

    return delta, gamma, vega, theta, rho
