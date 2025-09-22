# Implementation of the Black-Scholes Formula using Python

import numpy as np
from scipy.stats import norm # Normal distribution is used by the formula




def blackScholes(r: float, S: float, X: float, T: float, sigma: float, type: str) -> float:
    
    # r: Risk-Free Interest Rate
    # S: Underlying Stock Price ($)
    # X: Strike/Exercise Price ($)
    # T: Time to Expiration (in years)
    # sigma: Volatility (standard dev. of log returns)
    # type: defines the type of option -> "C" for Call and "P" for put

    # Calculating d1 and d2 (used by the formula)

    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    d2 = d1 - sigma*np.sqrt(T)
    
    if type == "C": # Call Option Case
        price = S*norm.cdf(d1, 0, 1) - X*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "P": # Put Option Case
        price = X*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    return price


# Calculating Delta Greek of European Option
def calcDelta(r: float, S: float, X: float, T: float, sigma: float, type: str) -> float:

    # Calculating d1
    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    
    if type == "C": # Call Option Case
        delta = norm.cdf(d1, 0, 1)
    elif type == "P": # Put Option Case
        delta = -norm.cdf(-d1, 0, 1)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    return delta

# Calculating Gamma Greek of European Option
def calcGamma(r: float, S: float, X: float, T: float, sigma: float) -> float:
    
    # Calculating d1
    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    
    gamma = norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))   
    return gamma

# Calculating Vega Greek of European Option
def calcVega(r: float, S: float, X: float, T: float, sigma: float) -> float:

    # Calculating d1 and d2
    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    
    vega = S*norm.pdf(d1, 0, 1)*np.sqrt(T)
    return vega/100 # Sensitivity to 1% change in IV (vega alone is respecive of a 100% change in IV)

# Calculating Theta Greek of European Option
def calcTheta(r: float, S: float, X: float, T: float, sigma: float, type: str) -> float:

    # Calculating d1 and d2
    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    d2 = d1 - sigma*np.sqrt(T)
    
    if type == "C": # Call Option Case
        theta = -S*norm.pdf(d1, 0, 1)*sigma/(2*T) - r*X*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "P": # Put Option Case
        theta = -S*norm.pdf(d1, 0, 1)*sigma/(2*T) + r*X*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    
    return theta/365 # Time decay per day (theta alone would be respective of a year: T is measured in years)

# Calculating Rho Greek of European Option
def calcRho(r: float, S: float, X: float, T: float, sigma: float, type: str) -> float:

    # Calculating d1 and d2
    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    d2 = d1 - sigma*np.sqrt(T)
    
    if type == "C": # Call Option Case
        rho = X*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "P": # Put Option Case
        rho = -X*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    else:
        raise ValueError("option_type must be 'C' for Call or 'P' for Put")
    
    return rho/100 # Sensitivity to 1% change in interest rates (rho alone is respective of a 100% change in interest rates)


# Calculate and Return all Greeks for Black Scholes Formula
def calcBSGreeks(r: float, S: float, X: float, T: float, sigma: float, type: str) -> tuple[float, float, float, float, float]:

    delta = calcDelta(r, S, X, T, sigma, type)
    gamma = calcGamma(r, S, X, T, sigma)
    vega = calcVega(r, S, X, T, sigma)
    theta = calcTheta(r, S, X, T, sigma, type)
    rho = calcRho(r, S, X, T, sigma, type)

    return delta, gamma, vega, theta, rho




