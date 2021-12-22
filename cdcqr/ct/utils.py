import pandas as pd
import matplotlib.pyplot as plt


def dt2ts(d,delta='1us'):
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        res=(d- pd.Timestamp("1970-01-01")) // pd.Timedelta(delta)
    else:
        res=(d- datetime.datetime(1970,1,1,tzinfo=timezone.utc)) // pd.Timedelta(delta)

    return res


def plot2(df,col1,col2,title='',figsize=(10,5),**kwargs):
    from cycler import cycler
    if type(col1)!=type([]):
        col1=[col1]
    if type(col2)!=type([]):
        col2=[col2]
    
    custom_cycler = (cycler(color=list('rbgcmk')))
    fig, ax1 = plt.subplots(figsize=figsize)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax1.set_ylabel(col1[0])
    ax1.set_prop_cycle(custom_cycler)
    for col in col1:
        ax1.plot(df.index, df[col],label=col,**kwargs)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx() 
    ax2.set_ylabel(col2[0])  
    ax1.set_prop_cycle(custom_cycler)
    for col in col2:
        ax2.plot(df.index, df[col],label=col,**kwargs)
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')
    fig.tight_layout()  
    plt.title(title)
    try:
        plt.text(0.9, 0.9, "pearson = {:.1%}".format(df[[col1,col2]].corr().iloc[0,1]), transform=plt.gca().transAxes, fontsize=15)
    except:
        pass
    plt.show()

pd.core.frame.DataFrame.plot2=plot2  


def rollingcorr(df, window, col=None, min_periods=None, method='spearman'):
    if method =='pearson':
        return df.rolling(window).corr(col)
    elif method=='spearman':
        return rolling_rank_corr(df, window, y = col, min_periods=1)
    else:
        raise ValueError(f"method {method} not implemented")

pd.core.frame.DataFrame.rollingcorr=rollingcorr