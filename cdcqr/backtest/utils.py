import sys
sys.path.append('/home/user/python-libs')
import backtrader as bt


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
