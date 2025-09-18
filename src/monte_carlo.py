import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from black_scholes import calcDelta as delta, calcGamma as gamma

# initial option parameters for implementation of Delta-Based Control Variates for Variance Rediction

S = 101.15 # stock price
X = 98.01 # Strike Price
vol = 0.0991 # volatility
r = 0.015 # risk-free rate
N = 20 # number of time steps
M = 1000 # number of simulations

market_value = 3.86 # market price of option

T = 60/365 # Time to expiry (in years)


# Helper functions

# Precomputation of Constants
def precompConst(S: float, vol: float, r: float, T: float, N: int) -> tuple[float, float, float, float]:
    # Length of single time step (in years): Time to Expiry divided into N steps
    dt = T/N
    
    # Drift term per time step: Representing the expected change (under the risk neutral measure) in ln(S) due to drift over each step
    drift_dt = (r - vol**2/2)*dt
    
    # Diffusion term per time step: Strength of Randomness in each time step
    volsdt = vol*np.sqrt(dt)

    # Natural Logarithm of Current Stock Price
    lnS = np.log(S)

    return drift_dt, volsdt, lnS, dt

# Computation of Expectation and Standard Error
def calcExpValAndSE(r: float, T: float, sum_payoff: float, sum_payoff2: float, M: int, discounted_payoff: np.ndarray, vectorized: bool) -> tuple[float, float]:
    
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
def stockPricePaths(lnSt: np.ndarray, M: int) -> None:

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


# Variants of Monte Carlo Simulator for (European) Options Pricing

# Slow Solution for Monte Carlo Implementation - Loops.
def monteCarloV1(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)
    # type: defines the type of option -> "C" for Call and "P" for put


    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)


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

        # Payoff is calculated differently for Call and Put options
        if type == "C": # Call Option Case
            payoff = max(0, ST - X) # Call Payoff: If ST > X, payoff is ST - X; otherwise payoff is 0 (option not exercised)
        elif type == "P": # Put Option Case
            payoff = max(0, X - ST) # Put Payoff: If X > ST, payoff is X - ST; otherwise payoff is 0 (option not exercised)
        else:
            raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
                
        # Accumulate this simulation's payoff (CT) and its square for expectation and variance calculation
        sum_payoff = sum_payoff + payoff
        sum_payoff2 = sum_payoff2 + payoff**2

    # Computing Expected Option Value and Standard Error
    C0, SE =  calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, 0, False)

    return C0, SE
    #stockPricePaths(lnSt, M)


# Implementing Variance Reductoin with Antithetic Variates for the Slow Solution.
def monteCarloV1A(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

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

        # Payoff is calculated differently for Call and Put options
        if type == "C": # Call Option Case
            payoff = (max(0, ST1 - X) + max(0, ST2 - X))/2 # Average Call Payoff of the two assets
        elif type == "P": # Put Option Case
            payoff = (max(0, X - ST1) + max(0, X - ST2))/2 # Average Put Payoff of the two assets
        else:
            raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
        
        # Accumulate this simulation's payoff and its square for expectation and variance calculation
        sum_payoff = sum_payoff + payoff
        sum_payoff2 = sum_payoff2 + payoff**2

        # Computing Expected Option Value and Standard Error
        C0, SE =  calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, 0, False)

        return C0, SE
        #stockPricePaths(lnSt, M)


# Implementing Variance Reductoin with (Delta-based) Control Variates for the Slow Solution.
def monteCarloV1DC(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    beta1 = -1  # Hardcoded beta coefficient for control variate adjustment; typically estimated but fixed here for simplicity

    # Standard Error Placeholders
    sum_payoff = 0 # Accumulator for the sum of simulated option payoffs
    sum_payoff2 = 0 # Accumulator for the sum of squared payoffs


    # Monte Carlo Method
    for i in range(M): # M simulations take place (each with N time steps)
        St = S
        cv = 0 # control variate accumulator

        # Simulation of N time steps (to expiry date)
        for j in range(N):

            epsilon = np.random.normal()

            # Calculate delta of the option at current step for control variate adjustment
            deltaSt = delta(r, S, X, T-j*dt, vol, type)

            # Simulate next stock price using geometric Brownian motion
            next_St = St*np.exp(drift_dt + volsdt*epsilon)

            # Accumulate control variate term: delta times difference between actual and expected increment
            cv = cv + deltaSt*(next_St - St*erdt)

            St = next_St

        # Adjust payoff by adding control variate term scaled by beta1
        # This reduces variance by leveraging correlation between payoff and control variate
        if type == "C":
            payoff = max(0, St - X) + beta1*cv
        elif type == "P":
            payoff = max(0, X - St) + beta1*cv
        else:
            raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")

        sum_payoff = sum_payoff + payoff
        sum_payoff2 = sum_payoff2 + payoff**2

    
    # Computing Expected Option Value and Standard Error
    # Note: payoff already includes control variate adjustment; discounting done in calcExpValAndSE
    C0, SE =  calcExpValAndSE(r, T, sum_payoff, sum_payoff2, M, 0, False)
    
    return C0, SE
    #stockPricePaths(lnSt, M)


# Vectorized Version - Faster, more efficient implementation of Monte Carlo Simulation.
def monteCarloV2(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)


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

    # Payoff is calculated differently for call and pay options
    if type == "C":
        discounted_payoff = np.exp(-r*T) * np.maximum(0, ST[-1] - X)
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * np.maximum(0, X - ST[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")

    # Computing Expected Option Value and Standard Error
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    return C0, SE
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with Antithetic Variates for the Vectorized Version.
def monteCarloV2A(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:

    # S: Stock Price ($)
    # X: Strike/Exercise Price ($)
    # vol: Volatility (%)
    # r: Risk-Free Interest Rate (%)
    # N: Number of Time Steps
    # M: Number of Simulations
    # T: Time to Expiry (in years)

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)


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

    # Payoff is calculated differently for call and pay options
    if type == "C":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, ST1[-1] - X) + np.maximum(0, ST2[-1] - X))/2
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, X - ST1[-1]) + np.maximum(0, X - ST2[-1]))/2
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    

    # Computing Expected Option Value and Standard Error
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    return C0, SE
    #stockPricePaths(lnSt, M)


# Implementing Variance Reductoin with (Delta-based) Control Variates for the Vectorized Solution.
def monteCarloV2DC(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:
    
    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    cv = 0
    beta1 = -1  # Hardcoded beta coefficient for control variate adjustment; fixed for simplicity

    # Monte Carlo Method

    # NxM matrix of standard normal random numbers: each cell represents the random increment (for each time step in each simulation)
    Z = np.random.normal(size = (N, M))
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St = drift_dt + volsdt*Z

    # Calculate stock price paths by cumulative product of exponentials of increments
    ST = S*np.cumprod(np.exp(delta_St), axis = 0)

    # Prepend initial stock price as first row to get full path including time 0
    ST = np.concatenate((np.full(shape = (1, M), fill_value = S), ST))

    # Create time grid for remaining time to expiry at each time step; avoid zero to prevent divide-by-zero in delta calculation
    # This grid goes from T down to dt (not zero)
    deltaST = delta(r, ST[:-1].T, X, np.linspace(T, dt, N), vol, type).T

    # Calculate cumulative sum of control variates: delta * (actual increment - expected increment)
    # This accumulates the control variate adjustment over all time steps for each simulation
    cv = np.cumsum(deltaST*(ST[1:] - ST[:-1]*erdt), axis = 0)
    

    # Payoff is calculated differently for call and pay options
    # Adjust payoff by adding scaled control variate (last cumulative value) before discounting
    if type == "C":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, ST[-1] - X) + beta1*cv[-1])
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, X - ST[-1]) + beta1*cv[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    

    # Computing Expected Option Value and Standard Error
    # Using vectorized discounted payoff array including control variate adjustment
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    return C0, SE
    #stockPricePaths(lnSt, M)


# Implementing Variance Reductoin with (Gamma-based) Control Variates for the Vectorized Solution.
def monteCarloV2GC(S: float, X: float, vol: float, r: float, N: int, M: int, T: float, type: str) -> tuple[float, float]:
    
    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    ergamma = np.exp((2*r + vol**2)*dt) - 2*erdt + 1

    beta2 = -0.5

    # Monte Carlo Method

    # NxM matrix of standard normal random numbers: each cell represents the random increment (for each time step in each simulation)
    Z = np.random.normal(size = (N, M))
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St = drift_dt + volsdt*Z

    # Calculate stock price paths by cumulative product of exponentials of increments
    ST = S*np.cumprod(np.exp(delta_St), axis = 0)

    # Prepend initial stock price as first row to get full path including time 0
    ST = np.concatenate((np.full(shape = (1, M), fill_value = S), ST))

    # Create time grid for remaining time to expiry at each time step; avoid zero to prevent divide-by-zero in delta calculation
    # This grid goes from T down to dt (not zero)
    gammaST = gamma(r, ST[:-1].T, X, np.linspace(T, dt, N), vol, type).T

    # Calculate cumulative sum of control variates: delta * (actual increment - expected increment)
    # This accumulates the control variate adjustment over all time steps for each simulation
    cv2 = np.cumsum(gammaST*((ST[1:] - ST[:-1])**2 - ergamma*ST[:-1]**2), axis = 0)
    

    # Payoff is calculated differently for call and pay options
    # Adjust payoff by adding scaled control variate (last cumulative value) before discounting
    if type == "C":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, ST[-1] - X) + beta2*cv2[-1])
    elif type == "P":
        discounted_payoff = np.exp(-r*T) * (np.maximum(0, X - ST[-1]) + beta2*cv2[-1])
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    

    # Computing Expected Option Value and Standard Error
    # Using vectorized discounted payoff array including control variate adjustment
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    return C0, SE
    #stockPricePaths(lnSt, M)


def comparisons(S, X, vol, r, N, M, T, type):
    
    C02, SE2 = monteCarloV2(S, X, vol, r, N, M, T, type)
    C03, SE3 = monteCarloV2A(S, X, vol, r, N, M, T, type)
    C04, SE4 = monteCarloV2DC(S, X, vol, r, N, M, T, type)
    C04, SE5 = monteCarloV2GC(S, X, vol, r, N, M, T, type)


    a = SE2/SE2
    b = SE2/SE3
    c = SE2/SE4
    d = SE2/SE5

    results = {
        "Vectorized Monte Carlo": {"SE": SE2, "SE Reduction Multiple": a},
        "Vectorized Monte Carlo with Antithetic Variates": {"SE": SE3, "SE Reduction Multiple": b},
        "Vectorized Monte Carlo with Delta-based Control Variates": {"SE": SE4, "SE Reduction Multiple": c},
        "Vectorized Monte Carlo with Gamma-based Control Variates": {"SE": SE5, "SE Reduction Multiple": d},

    }



    # Build DataFrame
    df = pd.DataFrame(results).T  # Transpose so versions are rows
    df.columns = ["Standard Error", "Standard Error Reduction Multiple"]

    print(df)

comparisons(S, X, vol, r, N, M, T, "C")