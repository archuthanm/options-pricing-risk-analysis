import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt

# Helper functions

# Precomputation of Constants
def precompConst(S, vol, r, T, N):
    # Length of single time step (in years): Time to Expiry divided into N steps
    dt = T/N
    
    # Drift term per time step: Representing the expected change (under the risk neutral measure) in ln(S) due to drift over each step
    drift_dt = (r - vol**2/2)*dt
    
    # Diffusion term per time step: Strength of Randomness in each time step
    volsdt = vol*np.sqrt(dt)

    # Natural Logarithm of Current Stock Price
    lnS = np.log(S)

    return drift_dt, volsdt, lnS

# Computation of Expectation and Standard Error
def calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, discounted_payoff, vectorized):
    
    if (not vectorized):
        # Expected Option Value: sum_CT/M = average payoff across all simulations; np.exp(-r*T) -> discounted back to present value under the risk-free rate.
        C0 = np.exp(-r*T)*sum_payoff/M
        
        # Discounted sample standard deviation of payoffs: measures spread of simulated option payoffs
        sigma = np.sqrt((sum_payoff2 - sum_payoff**2/M) * np.exp(-2*r*T) / (M - 1))
        
        # Standard Error: estimate of uncertainty in Monte Carlo option price estimate
        SE = sigma/np.sqrt(M)
    else :
        # Compute Monte Carlo estimate of option price: average of final discounted payoffs
        C0 = np.sum(discounted_payoff) / M
        
        # Sample standard deviation of final payoffs to measure spread
        sigma = np.sqrt(np.sum((discounted_payoff - C0)**2) / (M - 1))
        
        # Standard error of Monte Carlo estimate: sigma divided by sqrt of number of simulations
        SE = sigma/np.sqrt(M)

    return C0, SE

# Visualize stock price paths
def stockPricePaths(lnSt, M):

    ST_paths = np.exp(lnSt)
    num_paths_to_plot = M

    plt.figure(figsize=(10,6))
    for i in range(num_paths_to_plot):
        plt.plot(ST_paths[:, i], label=f'Path {i+1}')

    plt.xlabel('Time step')
    plt.ylabel('Stock Price')
    plt.title('Sample Monte Carlo Stock Price Paths')
    plt.grid(True)
    plt.show()


# Slow Solution for Monte Carlo Implementation - Loops.
def monteCarloV1(S, X, vol, r, N, M, T) :

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS = precompConst(S, vol, r, T, N)


    # Standard Error Placeholders
    sum_payoff = 0 # Accumulator for the sum of simulated option payoffs
    sum_payoff2 = 0 # Accumulator for the sum of squared payoffs

    # Monte Carlo Method
    for i in range(M): # M simulations take place (each with N time steps)
        lnSt = lnS

        # Simulation of N time steps (to expiry date)
        for j in range(N):
            lnSt = lnSt + drift_dt + volsdt*np.random.normal() # Update log-price: add deterministic drift and random diffusion
        
        ST = np.exp(lnSt) # Simulated Final Stock Price
        payoff = max(0, ST - X) # Payoff: If ST > X, payoff is ST - X; otherwise payoff is 0 (option not exercised)
        
        # Accumulate this simulation's payoff (CT) and its square for expectation and variance calculation
        sum_payoff = sum_payoff + payoff
        sum_payoff2 = sum_payoff2 + payoff**2

    # Computing Expected Option Value and Standard Error
    C0, SE =  calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, 0, False)

    print("V1: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))
    #stockPricePaths(lnSt, M)


# Implementing Variance Reductoin with Antithetic Variates for the Slow Solution.
def monteCarloV1A(S, X, vol, r, N, M, T) :

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS = precompConst(S, vol, r, T, N)

    # Standard Error Placeholders
    sum_payoff = 0 # Accumulator for the sum of simulated option payoffs
    sum_payoff2 = 0 # Accumulator for the sum of squared payoffs


    # Monte Carlo Method
    for i in range(M): # M simulations take place (each with N time steps)
        lnSt1 = lnS
        lnSt2 = lnS

        # Simulation of N time steps (to expiry date)
        for j in range(N):

            # Perfectly negatively correated assets: lnSt1 and lnSt2
            epsilon = np.random.normal()

            # Updating log-prices for both assets: add deterministic drift and random diffusion
            lnSt1 = lnSt1 + drift_dt + volsdt*epsilon
            lnSt2 = lnSt2 + drift_dt - volsdt*epsilon
        
        ST1 = np.exp(lnSt1) # Simulated Final Stock Price of asset 1
        ST2 = np.exp(lnSt2) # Simulated Final Stock Price of asset 2

        payoff = (max(0, ST1 - X) + max(0, ST2 - X))/2 # Average Payoff of the two assets.
        
        # Accumulate this simulation's payoff and its square for expectation and variance calculation
        sum_payoff = sum_payoff + payoff
        sum_payoff2 = sum_payoff2 + payoff**2

    
    # Computing Expected Option Value and Standard Error
    C0, SE =  calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, 0, False)
    
    print("V1A: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))
    #stockPricePaths(lnSt, M)


# Vectorized Version - Faster, more efficient implementation of Monte Carlo Simulation.
def monteCarloV2(S, X, vol, r, N, M, T):

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS = precompConst(S, vol, r, T, N)


    # Monte Carlo Method

    # NxM matrix of standard normal random numbers: each cell represents the random increment (for each time step in each simulation)
    Z = np.random.normal(size = (N, M))
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_lnst = drift_dt + volsdt*Z
    
    # NxM matrix of cumulative ln(S) paths: each column is one simulation, each row = ln(S) after that time step
    lnSt = lnS + np.cumsum(delta_lnst, axis = 0) # axis = 0 signifies to take cumulative sum down the columns
    
    # Prepend initial log-stock price as first row: resulting (N+1)xM matrix holds full ln(S) paths for all simulations
    lnSt = np.concatenate((np.full(shape = (1, M), fill_value = lnS), lnSt))


    # Convert ln-prices to actual stock prices for all time steps and simulations
    ST = np.exp(lnSt)

    # Discounted payoff of call option at each time step for all simulations (only last row is needed: for European Call options)
    discounted_payoff = np.exp(-r*T) * np.maximum(0, ST[-1] - X)

    
    # Computing Expected Option Value and Standard Error
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    print("V2: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with Antithetic Variates for the Vectorized Version.
def monteCarloV2A(S, X, vol, r, N, M, T):

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS = precompConst(S, vol, r, T, N)


    # Monte Carlo Method

    # NxM matrix of standard normal random numbers: each cell represents the random increment (for each time step in each simulation)
    Z = np.random.normal(size = (N, M))
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_lnSt1 = drift_dt + volsdt*Z
    delta_lnSt2 = drift_dt - volsdt*Z
    
    # NxM matrix of cumulative ln(S) paths: each column is one simulation, each row = ln(S) after that time step
    lnSt1 = lnS + np.cumsum(delta_lnSt1, axis = 0)
    lnSt2 = lnS + np.cumsum(delta_lnSt2, axis = 0)


    # Convert log-prices to actual stock prices for both (perfectly negatively correlated) assets
    ST1 = np.exp(lnSt1)
    ST2 = np.exp(lnSt2)

    # Discounted payoff of call option at each time step for all simulations (only last row is needed: for European Call options)
    discounted_payoff = np.exp(-r*T) * (np.maximum(0, ST1[-1] - X) + np.maximum(0, ST2[-1] - X))/2
    

    # Computing Expected Option Value and Standard Error
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    print("V2A: Call option value is ${0} with Standard Error +/- {1}".format(np.round(C0,2), np.round(SE,2)))
    #stockPricePaths(lnSt, M)