import numpy as np
import scipy.stats as si
from scipy.stats import norm


def euro_vanilla(S, K, T, r, sigma, option='call'):
    """
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return result


N = norm.cdf


def bs_call(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff / vega  # f(x) / f'(x)
    return sigma  # value wasn't found, return best guess so far


def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


'''
Input parameters:
S -> asset price
K -> strike price
r -> interest rate
sigma -> volatility
T -> time to maturity
'''


class Put:
    def Price(S, K, r, sigma, T):
        return np.maximum(K - S, 0) if T == 0 else K * np.exp(-r * T) * si.norm.cdf(
            -1 * d2(S, K, r, sigma, T)) - S * si.norm.cdf(-1 * d1(S, K, r, sigma, T))

    def Delta(S, K, r, sigma, T):
        return si.norm.cdf(d1(S, K, r, sigma, T)) - 1

    def Gamma(S, K, r, sigma, T):
        return si.norm.pdf(d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))

    def Vega(S, K, r, sigma, T):
        return S * si.norm.pdf(d1(S, K, r, sigma, T)) * np.sqrt(T)

    def Theta(S, K, r, sigma, T):
        aux1 = -S * si.norm.pdf(d1(S, K, r, sigma, T)) * sigma / (2 * np.sqrt(T))
        aux2 = r * K * np.exp(-r * T) * si.norm.cdf(-1 * d2(S, K, r, sigma, T))
        return aux1 + aux2

    def Rho(S, K, r, sigma, T):
        return -K * T * np.exp(-r * T) * si.norm.cdf(-1 * d2(S, K, r, sigma, T))

    def Get_range_value(Smin, Smax, Sstep, K, r, sigma, T, num_curves, value="Price"):
        vec = np.linspace(Smin, Smax, (Smax - Smin) / Sstep)
        vecT = np.linspace(0, T, num_curves, endpoint=True)
        if value == "Price":
            return vec, vecT, [[Put.Price(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value == "Delta":
            return vec, vecT, [[Put.Delta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value == "Gamma":
            return vec, vecT, [[Put.Gamma(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value == "Vega":
            return vec, vecT, [[Put.Vega(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value == "Theta":
            return vec, vecT, [[Put.Theta(S, K, r, sigma, t) for S in vec] for t in vecT]
        elif value == "Rho":
            return vec, vecT, [[Put.Rho(S, K, r, sigma, t) for S in vec] for t in vecT]
