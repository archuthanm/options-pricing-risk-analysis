import time
import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from black_scholes import calcDelta as delta, calcGamma as gamma, calcVega as vega, calcTheta as theta, calcRho as rho

# initial option parameters for implementation of Delta-Based Control Variates for Variance Rediction

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


def comparisons(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> None:
    
    C02, SE2, comp2 = monteCarloV2(S, X, vol, r, N, M, Z, T, type)
    C03, SE3, comp3 = monteCarloV2A(S, X, vol, r, N, M, Z, T, type)
    C04, SE4, comp4 = monteCarloV2DC(S, X, vol, r, N, M, Z, T, type)
    C05, SE5, comp5 = monteCarloV2GC(S, X, vol, r, N, M, Z, T, type)
    C06, SE6, comp6 = monteCarloV2AD(S, X, vol, r, N, M, Z, T, type)
    C07, SE7, comp7 = monteCarloV2Final(S, X, vol, r, N, M, Z, T, type)

    results = [
        {
            "Function": "Vectorized Monte Carlo",
            "Standard Error": SE2,
            "Computation Time": comp2,
            "Standard Error Reduction Multiple": SE2/SE2,
            "Relative Computation Time": comp2/comp2,
        },
        {
            "Function": "Vectorized Monte Carlo with Antithetic Variates",
            "Standard Error": SE3,
            "Computation Time": comp3,
            "Standard Error Reduction Multiple": SE2/SE3,
            "Relative Computation Time": comp3/comp2,
        },
        {
            "Function": "Vectorized Monte Carlo with Delta-based Control Variates",
            "Standard Error": SE4,
            "Computation Time": comp4,
            "Standard Error Reduction Multiple": SE2/SE4,
            "Relative Computation Time": comp4/comp2,
        },
        {
            "Function": "Vectorized Monte Carlo with Gamma-based Control Variates",
            "Standard Error": SE5,
            "Computation Time": comp5,
            "Standard Error Reduction Multiple": SE2/SE5,
            "Relative Computation Time": comp5/comp2,
        },
        {
            "Function": "Vectorized Monte Carlo with Antithetic AND Delta Variates",
            "Standard Error": SE6,
            "Computation Time": comp6,
            "Standard Error Reduction Multiple": SE2/SE6,
            "Relative Computation Time": comp6/comp2,
        },
        {
            "Function": "Vectorized Monte Carlo with Antithetic, Delta AND Gamma Variates",
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
    print(df)


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
def monteCarloV2(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:

    start_time = time.time()

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

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with Antithetic Variates for the Vectorized Version.
def monteCarloV2A(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:

    start_time = time.time()

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

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with (Delta-based) Control Variates for the Vectorized Solution.
def monteCarloV2DC(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:

    start_time = time.time()

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    cv = 0
    beta1 = -1  # Hardcoded beta coefficient for control variate adjustment; fixed for simplicity

    # Monte Carlo Method
    
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

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with (Gamma-based) Control Variates for the Vectorized Solution.
def monteCarloV2GC(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:
    
    start_time = time.time()

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    ergamma = np.exp((2*r + vol**2)*dt) - 2*erdt + 1

    beta2 = -0.5

    # Monte Carlo Method
    
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

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with the combination of Antithetic Variates AND (Delta-based) Control Variates for the Vectorized Solution.
def monteCarloV2AD(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:
    
    start_time = time.time()

    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    beta1 = -1  # Hardcoded beta coefficient for control variate adjustment; fixed for simplicity

    # Monte Carlo Method
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St1 = drift_dt + volsdt*Z

    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St2 = drift_dt - volsdt*Z
    

    # Calculate stock price paths by cumulative product of exponentials of increments
    ST1 = S*np.cumprod(np.exp(delta_St1), axis = 0)
    ST2 = S*np.cumprod(np.exp(delta_St2), axis = 0)

    # Prepend initial stock price as first row to get full path including time 0
    ST1 = np.concatenate((np.full(shape = (1, M), fill_value = S), ST1))
    ST2 = np.concatenate((np.full(shape = (1, M), fill_value = S), ST2))


    # Create time grid for remaining time to expiry at each time step; avoid zero to prevent divide-by-zero in delta calculation
    # This grid goes from T down to dt (not zero)
    deltaST1 = delta(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    deltaST2 = delta(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol, type).T


    # Calculate cumulative sum of control variates: delta * (actual increment - expected increment)
    # This accumulates the control variate adjustment over all time steps for each simulation
    cv1 = np.cumsum(deltaST1*(ST1[1:] - ST1[:-1]*erdt), axis = 0)
    cv2 = np.cumsum(deltaST2*(ST2[1:] - ST2[:-1]*erdt), axis = 0)


    # Payoff is calculated differently for call and pay options
    # Adjust payoff by adding scaled control variate (last cumulative value) before discounting
    if type == "C":
        discounted_payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, ST1[-1] - X) + beta1*cv1[-1] + 
                                                    np.maximum(0, ST2[-1] - X) + beta1*cv2[-1]  )
    elif type == "P":
        discounted_payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, X - ST1[-1]) + beta1*cv1[-1] + 
                                                    np.maximum(0, X - ST2[-1]) + beta1*cv2[-1]  )
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    

    # Computing Expected Option Value and Standard Error
    # Using vectorized discounted payoff array including control variate adjustment
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Implementing Variance Reduction with the combination of Antithetic Variates, (Delta-based AND Gamma-Based) Control Variates for the Vectorized Solution.
def monteCarloV2Final(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:
    
    start_time = time.time()
    
    #Precompute Constants
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)

    # Exponential of (risk-free rate x change in time): forward factor
    erdt = np.exp(r*dt)

    ergamma = np.exp((2*r + vol**2)*dt) - 2*erdt + 1

    beta1 = -1  # Hardcoded beta coefficient for control variate adjustment; fixed for simplicity

    beta2 = -0.5

    # Monte Carlo Method
    
    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St1 = drift_dt + volsdt*Z

    # NxM matrix of ln(S) increments: drift + stochastic term (for each time step in each simulation)
    delta_St2 = drift_dt - volsdt*Z
    

    # Calculate stock price paths by cumulative product of exponentials of increments
    ST1 = S*np.cumprod(np.exp(delta_St1), axis = 0)
    ST2 = S*np.cumprod(np.exp(delta_St2), axis = 0)

    # Prepend initial stock price as first row to get full path including time 0
    ST1 = np.concatenate((np.full(shape = (1, M), fill_value = S), ST1))
    ST2 = np.concatenate((np.full(shape = (1, M), fill_value = S), ST2))


    # Create time grid for remaining time to expiry at each time step; avoid zero to prevent divide-by-zero in delta calculation
    # This grid goes from T down to dt (not zero)
    deltaST1 = delta(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    deltaST2 = delta(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol, type).T

    gammaST1 = gamma(r, ST1[:-1].T, X, np.linspace(T, dt, N), vol, type).T
    gammaST2 = gamma(r, ST2[:-1].T, X, np.linspace(T, dt, N), vol, type).T


    # Calculate cumulative sum of control variates: delta * (actual increment - expected increment)
    # This accumulates the control variate adjustment over all time steps for each simulation
    cv1d = np.cumsum(deltaST1*(ST1[1:] - ST1[:-1]*erdt), axis = 0)
    cv2d = np.cumsum(deltaST2*(ST2[1:] - ST2[:-1]*erdt), axis = 0)

    cv1g = np.cumsum(gammaST1*((ST1[1:] - ST1[:-1])**2 - ergamma*ST1[:-1]**2), axis = 0)
    cv2g = np.cumsum(gammaST2*((ST2[1:] - ST2[:-1])**2 - ergamma*ST2[:-1]**2), axis = 0)


    # Payoff is calculated differently for call and pay options
    # Adjust payoff by adding scaled control variate (last cumulative value) before discounting
    if type == "C":
        discounted_payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, ST1[-1] - X) + beta1*cv1d[-1] + beta2*cv1g[-1] +
                                                    np.maximum(0, ST2[-1] - X) + beta1*cv2d[-1] + beta2*cv2g[-1] )
    elif type == "P":
        discounted_payoff = 0.5 * np.exp(-r*T) * (  np.maximum(0, X - ST1[-1]) + beta1*cv1d[-1] + beta2*cv1g[-1] + 
                                                    np.maximum(0, X - ST2[-1]) + beta1*cv2d[-1] + beta2*cv2g[-1] )
    else:
        raise ValueError(f"Invalid option type '{type}'. Must be 'C' for Call or 'P' for Put.")
    

    # Computing Expected Option Value and Standard Error
    # Using vectorized discounted payoff array including control variate adjustment
    C0, SE = calcExpValAndSE(r, T, 0, 0, M, discounted_payoff, True)

    computation_time = time.time() - start_time

    return C0, SE, computation_time
    #stockPricePaths(lnSt, M)


# Calculating Greeks From Monte Carlo Simulator

def calcMCDelta(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str) -> tuple[float, float]:
    
    drift_dt, volsdt, lnS, dt = precompConst(S, vol, r, T, N)
    erdt = np.exp(r*dt)
    ergamma = np.exp((2*r + vol**2)*dt) - 2*erdt + 1
    beta1 = -1.0
    beta2 = -0.5

    # Antithetic log-price increments
    delta_St1 = drift_dt + volsdt*Z
    delta_St2 = drift_dt - volsdt*Z

    # Stock price paths
    ST1 = S * np.cumprod(np.exp(delta_St1), axis=0)
    ST2 = S * np.cumprod(np.exp(delta_St2), axis=0)
    ST1 = np.concatenate((np.full((1, M), S), ST1))
    ST2 = np.concatenate((np.full((1, M), S), ST2))

    # Time grid for remaining time to expiry
    t_grid = np.linspace(T, dt, N)

    # Delta and gamma along paths
    deltaST1 = delta(r, ST1[:-1].T, X, t_grid, vol, type).T
    deltaST2 = delta(r, ST2[:-1].T, X, t_grid, vol, type).T
    gammaST1 = gamma(r, ST1[:-1].T, X, t_grid, vol, type).T
    gammaST2 = gamma(r, ST2[:-1].T, X, t_grid, vol, type).T

    # Control variates cumulative sum
    cv1d = np.cumsum(deltaST1*(ST1[1:] - ST1[:-1]*erdt), axis=0)
    cv2d = np.cumsum(deltaST2*(ST2[1:] - ST2[:-1]*erdt), axis=0)
    cv1g = np.cumsum(gammaST1*((ST1[1:] - ST1[:-1])**2 - ergamma*ST1[:-1]**2), axis=0)
    cv2g = np.cumsum(gammaST2*((ST2[1:] - ST2[:-1])**2 - ergamma*ST2[:-1]**2), axis=0)

    # Pathwise delta at t=0 using chain rule adjustment
    if type == "C":
        delta_paths = 0.5 * ((ST1[-1] > X).astype(float) * ST1[-1]/S + beta1*cv1d[-1]/S + beta2*cv1g[-1]/S 
                            +(ST2[-1] > X).astype(float) * ST2[-1]/S + beta1*cv2d[-1]/S + beta2*cv2g[-1]/S)
    elif type == "P":
        delta_paths = 0.5 * (-(ST1[-1] < X).astype(float) * ST1[-1]/S - beta1*cv1d[-1]/S - beta2*cv1g[-1]/S
                             -(ST2[-1] < X).astype(float) * ST2[-1]/S - beta1*cv2d[-1]/S - beta2*cv2g[-1]/S)
    else:
        raise ValueError("type must be 'C' or 'P'")

    # Discounting back
    delta_paths *= np.exp(-r*T)

    delta_exp = np.mean(delta_paths)
    SE = np.std(delta_paths, ddof=1) / np.sqrt(M)

    return float(delta_exp), float(SE)

# Monte Carlo Gamma using Finite Difference (central difference) method, with proper SE propagation from the three bumped price estimates.
def calcMCGamma(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.5) -> tuple[float, float]:

    # Price at S+h
    C1, SE1, comp1 = monteCarloV2Final(S + h, X, vol, r, N, M, Z, T, type)
    
    # Price at S
    C2, SE2, comp2 = monteCarloV2Final(S, X, vol, r, N, M, Z, T, type)
    
    # Price at S-h
    C3, SE3, comp3 = monteCarloV2Final(S - h, X, vol, r, N, M, Z, T, type)
    
    # Central finite difference
    gamma_exp = (C1 - 2*C2 + C3) / (h**2)
    
    # Propagate SEs (assuming independence)
    SE = np.sqrt(SE1**2 + 4*SE2**2 + SE3**2) / (h**2)
    
    return float(gamma_exp), float(SE)

# Monte Carlo Vega using Finite Difference (central difference) method, with proper SE propagation.
def calcMCVega(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.01) -> tuple[float, float]:
    
    # Price at vol+h
    C1, SE1, comp1 = monteCarloV2Final(S, X, vol + h, r, N, M, Z, T, type)
    
    # Price at vol-h
    C2, SE2, comp2 = monteCarloV2Final(S, X, vol - h, r, N, M, Z, T, type)
    
    # Central finite difference for Vega
    vega_exp = (C1 - C2) / (2 * h)
    
    # Standard error propagation (assuming independence)
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / (2 * h)
    
    return float(vega_exp/100), float(SE/100)

# Monte Carlo Theta using Finite Difference (central difference) method, with proper SE propagation from the two bumped time estimates.
def calcMCTheta(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 1/365) -> tuple[float, float]:
    
    # Price at T+h_T
    C2, SE2, comp1 = monteCarloV2Final(S, X, vol, r, N, M, Z, T+h, type)
    
    # Price at T-h_T (ensure T-h_T > 0)
    if T - h <= 0:
        raise ValueError("T - h_T must be positive for finite difference Theta calculation.")
    C1, SE1, comp2 = monteCarloV2Final(S, X, vol, r, N, M, Z, T - h, type)
    
    # Central finite difference for Theta (per year)
    theta_exp = (C1 - C2) / h
    
    # Standard error propagation (assuming independence)
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / h
    
    return float(theta_exp/365), float(SE/365)

# Monte Carlo Rho using Finite Difference (central difference) method, with proper SE propagation from the two bumped rate estimates.
def calcMCRho(S: float, X: float, vol: float, r: float, N: int, M: int, Z: np.ndarray, T: float, type: str, h: float = 0.01) -> tuple[float, float]:
    
    # Price at r+h_r
    C1, SE1, comp1 = monteCarloV2Final(S, X, vol, r + h, N, M, Z, T, type)
    
    # Price at r-h_r
    C2, SE2, comp2 = monteCarloV2Final(S, X, vol, r - h, N, M, Z, T, type)
    
    # Central finite difference for Rho (per unit of r)
    rho_exp = (C1 - C2) / (2 * h)
    
    # Standard error propagation (assuming independence)
    SE = np.sqrt(SE1 ** 2 + SE2 ** 2) / (2 * h)
    
    return float(rho_exp/100), float(SE/100)

comparisons(S, X, vol, r, N, M, Z, T, "C")


print(calcMCDelta(S, X, vol, r, N, M, Z, T, "C"))

print(delta(r, S, X, T, vol, "C"))


print(calcMCGamma(S, X, vol, r, N, M, Z, T, "C"))

print(gamma(r, S, X, T, vol, "C"))


print(calcMCVega(S, X, vol, r, N, M, Z, T, "C"))

print(vega(r, S, X, T, vol, "C"))


print(calcMCTheta(S, X, vol, r, N, M, Z, T, "C"))

print(theta(r, S, X, T, vol, "C"))


print(calcMCRho(S, X, vol, r, N, M, Z, T, "C"))

print(rho(r, S, X, T, vol, "C"))