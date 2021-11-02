import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as si
from functools import partial
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm
from cdcqr.common.config import LOCAL_FIGURE_DIR
import os
import time
from datetime import datetime


def euro_vanilla(S, K, T, r, sigma, option='call'):
    """
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    # return option price
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


def volfit(df, plot=False, model='errfunc_parabolic_linear2', **kwargs):
    # res={}
    figs = []
    ts = df.index.get_level_values('tm').drop_duplicates()
    # ts=ts[:Nts] if Nts else ts
    #    ipdb.set_trace()
    res = []
    for t in ts:
        dft = df.loc[[t]].droplevel('tm')
        Ts = dft.index.get_level_values('expire').drop_duplicates()
        #   Ts=Ts[:NTs] if NTs else Ts
        for T in Ts:
            dftT = dft.loc[[T]].droplevel('expire').drop_duplicates()

            tau = (T - t) / pd.to_timedelta(1, unit='D') / 365
            dftT['logKFtau'] = np.log(dftT.index / dftT['s']) / np.sqrt(tau)
            S = dftT['s'].iloc[0]
            dftTOTM = dftT[((dftT.ty == 1) & (S < dftT.index)) | ((dftT.ty == -1) & (S > dftT.index))]
            if len(dftTOTM) == 0:
                continue
            # after this can be empy cuz all ITM
            print(f"T={T} t={t}")
            # display(dft.loc[[T]])
            # display(dftTOTM)
            # dftTOTM[['aiv','biv']].plot(title=f' t={t} T={T} S={S}',style='o')
            usevega = True
            # lookup initvals from res df , if exist
            prevparams = [0.2, 0.2, 0.2, -0.2] if model[-1] == '4' else [0.2] * int(model[-1])
            print(f"prevparams={prevparams} model={model}")

            minres = minimize(
                partial(eval(model), usevega=usevega, df=dftTOTM, PRICECALL=False, stats=False, plot=False,
                        kleft=kwargs['kleft'], kright=kwargs['kright']), prevparams, method='L-BFGS-B',
                options={'eps': 1e-09, 'maxfun': 100, 'maxiter': 100})
            # minres = minimize(partial(errfunc_parabolic_linear1,usevega=usevega,df=dftTOTM,PRICECALL=False,plot=False,kleft=-2,kright=2), [0.2,0.2,0.01,0.01,0.2,-0.2], method='L-BFGS-B', options={'eps': 1e-08, 'maxfun': 100, 'maxiter': 100})
            # TODO: put previous params as initial guess fot t+1, same T
            print(f"optimized params:{minres.x}")
            print(type(t))
            dt = pd.to_datetime(t)
            expiry_date = kwargs['expiry_date']
            fig_name = '{}_{}'.format(dt.strftime("%Y%m%d%H%M%S"), expiry_date)
            plotres = partial(eval(model), usevega=usevega, df=dftTOTM, PRICECALL=True, plot=plot, info=f't={t} T={T}',
                              kleft=kwargs['kleft'], kright=kwargs['kright'], fig_name=fig_name)(minres.x)
            if plot:
                figs.extend(plotres['figs'])
            resd = {'t': t, 'T': T, 'volATM': plotres['volATM'], 's': S, 'rr': plotres['rr'], 'conv': plotres['conv'],
                    'perc': plotres['perc']}
            for i in range(len(minres.x)):
                resd['p' + str(i)] = minres.x[i]
            res.append(resd)
    return res


def errfunc_parabolic_linear6(x, usevega=False, useweights=False, df=None, PRICECALL=False, plot=False, info="",
                              **kwargs):  # left skew and right skew - used in production
    # figsflask = not isnotebook()
    res = {}
    figs = []
    skewleft = x[0]
    skewright = x[1]
    leftconv = x[2]
    rightconv = x[3]
    rightwing = x[4]
    leftwing = x[5]

    # same skew ATM
    # skewleft=skewright

    # same conv ATM
    # leftconv=rightconv

    df['m'] = 0.5 * (df['aiv'] + df['biv'])
    S = df['s'].iloc[0]
    try:
        volATM = interp1d(df.index, df['m'])([S])[0]
    except Exception as e:
        print(str(e))
        print('dfindex', df.index)
        print('dfm', df['m'])
        resprice = {'rr': np.nan, 'conv': np.nan, 'volATM': np.nan, 'err': str(e)}
        if PRICECALL:
            if plot > 2:
                img = plotstart()
                plt.figure(figsize=(4, 4))
                plt.text(0.1, 0.9, f'{str(e)} {info}')
                resprice['figs'] = [plotend(img)]
            return resprice
        return 0
    # display(df)
    kright = kwargs['kright']
    kleft = kwargs['kleft']
    df.loc[(df.logKFtau > 0) & (df.logKFtau < kright), 'myvol'] = volATM + (
            skewright * df['logKFtau'] + rightconv * df['logKFtau'] ** 2)
    df.loc[(df.logKFtau < 0) & (df.logKFtau > kleft), 'myvol'] = volATM + (
            skewleft * df['logKFtau'] + leftconv * df['logKFtau'] ** 2)
    yright = volATM + (skewright * kright + rightconv * kright ** 2)
    yleft = volATM + (skewleft * kleft + leftconv * kleft ** 2)
    df.loc[(df.logKFtau > kright), 'myvol'] = rightwing * (df['logKFtau'] - kright) + yright
    df.loc[(df.logKFtau < kleft), 'myvol'] = leftwing * (df['logKFtau'] - kleft) + yleft
    df['insidebidask'] = False
    df.loc[(df.myvol > df.biv) & (df.myvol < df.aiv), 'insidebidask'] = True
    if PRICECALL and plot > 2:
        # #         df[['aiv','biv','myvol','markiv']].plot(title=f"{info} s={S} {df['insidebidask'].sum()/len(df):.1%} vols in bidask " ,style='.-')
        # #         plt.show()
        #         plt.figure(figsize=(5,5))
        df[['aiv', 'biv', 'myvol', 'markiv', 'logKFtau']].query("abs(logKFtau)<1").set_index('logKFtau').plot(
            title=f"{info} s={S} {df['insidebidask'].sum() / len(df):.1%} vols in bidask ", style='.-')  # remove markiv
        fig_name = kwargs.get('fig_name')
        plt.savefig(os.path.join(LOCAL_FIGURE_DIR, 'volfit', '{}.png'.format(fig_name)))
        plt.show()
        
        
    res = (df['myvol'] - df['m']) ** 2
    if usevega:
        res *= df['vega']

        if useweights:
            ws = np.ones(len(df))
            df['w'] = np.ones(len(df))
            df.loc[df['logKFtau'] > 2, 'w'] = 0
            df.loc[df['logKFtau'] < -2, 'w'] = 0
            res *= df['w']
    if PRICECALL:
        resprice = {'volATM': volATM}
        resprice['perc'] = df['insidebidask'].sum() / len(df)
        try:
            volp5 = interp1d(df.index, df['myvol'])([S * (1.05)])[0]
            volm5 = interp1d(df.index, df['myvol'])([S * (0.95)])[0]
            skew = volp5 - volm5  # risk reversal 5%
            conv = 0.5 * (volp5 + volm5) - volATM  # butterfly
            resprice['rr'] = skew
            resprice['conv'] = conv
        except Exception as e:
            resprice['rr'] = np.nan
            resprice['conv'] = np.nan
            print(f"!volfit {str(e)}")

        if plot:
            resprice['figs'] = figs
        return resprice
    return res.sum()
