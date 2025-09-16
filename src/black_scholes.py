# Implementation of the Black-Scholes Formula using Python

import numpy as np
from scipy.stats import norm # Normal distribution is used by the formula

def blackScholes(r, S, X, T, sigma, type): 
    
    # r: Risk-Free Interest Rate
    # S: Underlying Stock Price ($)
    # X: Strike/Exercise Price ($)
    # T: Time to Expiration (in years)
    # sigma: Volatility (standard dev. of log returns)
    # type: defines the type of option -> "C" for Call and "P" for put

    # Calculating d1 and d2 (used by the formula)

    d1 = ( np.log(S/X) + (r + sigma**2/2) * T ) / ( sigma*np.sqrt(T) )
    d2 = d1 - sigma*np.sqrt(T)
    
    try:
        if type == "C": # Call Option Case
            price = S*norm.cdf(d1, 0, 1) - X*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "P": # Put Option Case
            price = X*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        
        return price
    except:
        print("Error. Please confirm all option parameters were entered correctly and try again.")