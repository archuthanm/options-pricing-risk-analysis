from black_scholes import calcBSGreeks as bsGreeks
from monte_carlo import monteCarloV2A as mc # Vectorized Monte Carlo Options Pricing Simulator using Antithetic Variates

import numpy as np
from scipy.stats import norm

# r: Risk-Free Interest Rate
r = 0.01

# S: Underlying Stock Price ($)
S = 30

# X: Strike/Exercise Price ($)
X = 40

# T: Time to Expiration (in years)
T = 240/365

# sigma: Volatility (standard dev. of log returns)
sigma = 0.30


# Calculating Greeks for Black Scholes Options Pricing Formula
delta, gamma, vega, theta, rho = bsGreeks(r, S, X, T, sigma, "C")

