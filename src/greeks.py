from black_scholes import calcBSGreeks as bsGreeks
from monte_carlo import calcMCGreeks as mc # Vectorized Monte Carlo Options Pricing Simulator using Antithetic and Control Variates

import numpy as np
import pandas as pd
from scipy.stats import norm

S = 101.15 # stock price
X = 98.01 # Strike Price
vol = 0.0991 # volatility
r = 0.015 # risk-free rate
N = 20 # number of time steps
M = 1000 # number of simulations

market_value = 3.86 # market price of option
T = 60/365 # Time to expiry (in years)

# NxM matrix of standard normal random numbers: each cell represents the random increment (for each time step in each simulation)
Z = np.random.normal(size = (N, M))


def greeksComparison(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> pd.DataFrame:
    
    # Calculating Greeks for Black Scholes Options Pricing Formula
    BSdelta, BSgamma, BSvega, BStheta, BSrho = bsGreeks(r, S, X, T, vol, type)

    # Calculating Greeks for MC Options Pricing Simulator
    MCdelta, SEdelta, MCgamma, SEgamma, MCvega, SEvega, MCtheta, SEtheta, MCrho, SErho = mc(S, X, vol, r, N, M, Z, T, type)

    results = [
            {
                "Greek": "Delta",
                "Black-Scholes Formula": BSdelta,
                "Monte Carlo Simulator": MCdelta,
                "Standard Error from Monte Carlo Simulator": SEdelta,
            },
            {
                "Greek": "Gamma",
                "Black-Scholes Formula": BSgamma,
                "Monte Carlo Simulator": MCgamma,
                "Standard Error from Monte Carlo Simulator": SEgamma,
            },
            {
                "Greek": "Vega",
                "Black-Scholes Formula": BSvega,
                "Monte Carlo Simulator": MCvega,
                "Standard Error from Monte Carlo Simulator": SEvega,
            },
            {
                "Greek": "Theta",
                "Black-Scholes Formula": BStheta,
                "Monte Carlo Simulator": MCtheta,
                "Standard Error from Monte Carlo Simulator": SEtheta,
            },
            {
                "Greek": "Rho",
                "Black-Scholes Formula": BSrho,
                "Monte Carlo Simulator": MCrho,
                "Standard Error from Monte Carlo Simulator": SErho,
            }
        ]
    pd.set_option('display.max_colwidth', None)

    columns = [
            "Greek",
            "Black-Scholes Formula",
            "Monte Carlo Simulator",
            "Standard Error from Monte Carlo Simulator",
        ]

    df = pd.DataFrame(results, columns=columns)
    df = df.round(3)
    return df

print(greeksComparison(S, X, vol, r, N, M, Z, T, "C"))