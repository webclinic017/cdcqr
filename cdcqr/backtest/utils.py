import sys
sys.path.append('/home/user/python-libs')
import backtrader as bt
import pandas as pd
import numpy as np


def initcerebro(icap=10000, fees=0.0006,multi=False,futures=False,margin=0.01,mult=1.0):
    cerebro = bt.Cerebro(stdstats=False)
    
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Trades)
    if multi:
        cerebro.addobservermulti(bt.observers.BuySell)    
    else:
        cerebro.addobserver(bt.observers.BuySell)
    percslippage=0# 10bps
    cerebro.broker.setcash(icap)
    if futures:
        cerebro.broker.setcommission(commission=fees,margin=margin,mult=mult)
    else:
        cerebro.broker.setcommission(commission=fees)
    
    cerebro.broker.set_slippage_perc(percslippage, slip_open=True, slip_limit=True, slip_match=True, slip_out=False)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(trade_list,_name='tradelist')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='txn')
    cerebro.addanalyzer(bt.analyzers.TimeReturn,_name='timereturn')
    #cerebro.addanalyzer(bt.analyzers.Returns,_name='returns')
    cerebro.addanalyzer(bt.analyzers.LogReturnsRolling,_name='logreturnsrolling')


    return cerebro



def diagnostic(df):
    num_days = (df.index[-1] - df.index[0]).days
    num_trades = (df['trade']!=0).sum()
    avg_num_trades_per_day =  round(num_trades/num_days, 4)
    avg_alpha_bps = round((df['raw_pnl'].sum()/num_trades)/df['c'].mean()*1e4, 4)
    avg_cost_bps = round((df['tc'].sum()/num_trades)/df['c'].mean()*1e4, 4)
    cost2alpha = round(avg_cost_bps/avg_alpha_bps, 4)
    
    df1d = df.resample('1D').agg({'pnl':'sum','raw_pnl':'sum','tc':'sum'})
    io = round(np.sqrt(365)*df1d['pnl'].mean()/df1d['pnl'].std(),4)
    io_raw = round(np.sqrt(365)*df1d['raw_pnl'].mean()/df1d['pnl'].std(),4)
    
    ret_dict = {}
    ret_dict['num_days'] = num_days
    ret_dict['num_trades'] = num_trades
    ret_dict['avg_num_trades_per_day'] = avg_num_trades_per_day
    ret_dict['avg_alpha_bps'] = avg_alpha_bps
    ret_dict['avg_cost_bps'] = avg_cost_bps
    ret_dict['cost2alpha'] = cost2alpha
    ret_dict['io'] = io
    ret_dict['io_raw'] = io_raw
    
    return ret_dict


def pos2pnl(df, fees=0.0007):
    df['trade'] = df['share_pos'].diff().fillna(0)
    df['dollar_pos'] = df['share_pos']*df['c']
    df['r'] = df['c'].pct_change() 
    df['raw_pnl'] = (df['dollar_pos'].shift(1)*df['r']).fillna(0)
    df['tc'] = (df['trade']*df['c']).abs()*fees
    df['pnl'] = df['raw_pnl'] - df['tc']
    return  diagnostic(df)


def signal_backtester_light(price, signal, n=200, fees=0.0007, lag=1):
    df = price.copy().to_frame()
    df.columns = ['c']
    df['signal'] = signal
    df['share_pos'] = df['signal'].shift(lag).replace(0, np.nan).ffill(limit=n).fillna(0)
    df['share_pos'][-1] = 0
    
    return pos2pnl(df, fees=fees)

