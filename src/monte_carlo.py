import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt

# Stock Price ($)
S = 101.15

# Strike/Exercise Price ($)
X = 98.01

# Volatility (%)
vol = 0.0991

# Risk-Free Interest Rate (%)
r = 0.01

# Number of Time Steps
N = 1

# Number of Simulations
M = 1000

# Market Price of Option
market_value = 3.86

# Time to Expiry (in years)
# Expiry Date: 17/03/2022
# Today's Date: 17/01/2022
T = ((datetime.date(2022,3,17) - datetime.date(2022, 1, 17)).days + 1)/365



def monteCarloV1(S, X, vol, r, N, M, T) :

    # Precomputation of Constants
    dt = T/N
    nudt = (r - vol**2/2)*dt
    volsdt = vol*np.sqrt(dt)
    lnS = np.log(S)

    # Standard Error Placeholders
    sum_CT = 0
    sum_CT2 = 0

    # Monte Carlo Method
    for i in range(M):
        lnSt = lnS
        for j in range(N):
            lnSt = lnSt + nudt + volsdt*np.random.normal()
        
        ST = np.exp(lnSt)
        CT = max(0, ST - X)
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 + CT**2

    # Computation of Expectation and Standard Error
    C0 = np.exp(-r*T)*sum_CT/M
    sigma = np.sqrt((sum_CT2 - sum_CT**2/M) * np.exp(-2*r*T) / (M - 1))
    SE = sigma/np.sqrt(M)

    print("V1: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))



def monteCarloV2(S, X, vol, r, N, M, T):

    # Precomputation of Constants
    dt = T/N
    nudt = (r - vol**2/2)*dt
    volsdt = vol*np.sqrt(dt)
    lnS = np.log(S)

    # Monte Carlo Method
    Z = np.random.normal(size = (N, M))
    delta_lnst = nudt + volsdt*Z
    lnSt = lnS + np.cumsum(delta_lnst, axis = 0)
    lnSt = np.concatenate((np.full(shape = (1, M), fill_value = lnS), lnSt))

    # Compute Expectation and Standard Error
    ST = np.exp(lnSt)
    CT = np.maximum(0, ST - X)
    C0 = np.exp(-r*T)*np.sum(CT[-1]) / M

    sigma = np.sqrt(np.sum((CT[-1] - C0)**2) / (M - 1))
    SE = sigma/np.sqrt(M)

    print("V2: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))


monteCarloV1(S, X, vol, r, N, M, T)
monteCarloV2(S, X, vol, r, N, M, T)