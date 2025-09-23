from black_scholes import compute_greeks as compute_greeks_bs
from monte_carlo import compute_greeks as compute_greeks_mc  # Vectorized Monte Carlo Options Pricing Simulator using Antithetic and Control Variates

import numpy as np
import pandas as pd
from scipy.stats import norm

S = 101.15  # Underlying asset price
X = 98.01  # Strike price
vol = 0.0991  # Volatility
r = 0.015  # Risk-free interest rate
N = 20  # Number of time steps
M = 1000  # Number of simulations

market_value = 3.86  # Market price of the option
T = 60 / 365  # Time to expiry in years

# Standard normal random matrix of shape (N, M), representing random increments for each step and simulation
Z = np.random.normal(size=(N, M))


def compare_greeks(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> pd.DataFrame:
    """
    Compare option Greeks computed using Black-Scholes formula and Monte Carlo simulation.

    Parameters:
        S (float): Underlying asset price
        X (float): Strike price
        vol (float): Volatility
        r (float): Risk-free rate
        N (int): Number of time steps
        M (int): Number of simulations
        Z (np.ndarray): Matrix of random normal variates
        T (float): Time to expiry in years
        type (str): Option type ('C' for Call, 'P' for Put)

    Returns:
        pd.DataFrame: DataFrame summarizing Greeks from both methods and Monte Carlo standard errors
    """
    # Calculate Greeks using Black-Scholes formula
    BSdelta, BSgamma, BSvega, BStheta, BSrho = compute_greeks_bs(r, S, X, T, vol, type)

    # Calculate Greeks and standard errors using Monte Carlo simulation
    MCdelta, SEdelta, MCgamma, SEgamma, MCvega, SEvega, MCtheta, SEtheta, MCrho, SErho = compute_greeks_mc(
        S, X, vol, r, N, M, Z, T, type
    )

    results = [
        {
            "Greek": "Delta",
            "Black-Scholes": BSdelta,
            "Monte Carlo": MCdelta,
            "MC SE": SEdelta,
        },
        {
            "Greek": "Gamma",
            "Black-Scholes": BSgamma,
            "Monte Carlo": MCgamma,
            "MC SE": SEgamma,
        },
        {
            "Greek": "Vega",
            "Black-Scholes": BSvega,
            "Monte Carlo": MCvega,
            "MC SE": SEvega,
        },
        {
            "Greek": "Theta",
            "Black-Scholes": BStheta,
            "Monte Carlo": MCtheta,
            "MC SE": SEtheta,
        },
        {
            "Greek": "Rho",
            "Black-Scholes": BSrho,
            "Monte Carlo": MCrho,
            "MC SE": SErho,
        },
    ]

    columns = ["Greek", "Black-Scholes", "Monte Carlo", "MC SE"]
    df = pd.DataFrame(results, columns=columns).round(3)
    pd.set_option('display.max_colwidth', None)
    return df


print(compare_greeks(S, X, vol, r, N, M, Z, T, "C"))


