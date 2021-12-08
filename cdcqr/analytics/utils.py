import matplotlib.pyplot as plt
from ct.utils import plot2, rollingcorr
import pandas as pd


def ts_eda(dff):
    # double y plot
    dff.plot2('v','y')
    
    # return scatter plots
    ret = dff.pct_change()
    lower_bound1 = ret['v'].quantile(0.05)
    upper_bound1 = ret['v'].quantile(0.95)
    lower_bound2 = ret['v'].quantile(0.001)
    upper_bound2 = ret['v'].quantile(0.999)
    
    ret_robust = ret[(ret['v']<=upper_bound1) & (ret['v']>=lower_bound1)]
    ret_outlier = ret[((ret['v']>upper_bound1) & (ret['v']<upper_bound2)) | ((ret['v']<lower_bound1) & (ret['v']>lower_bound2))]
    bad_data = ret[(ret['v']>=upper_bound2) | (ret['v']<=lower_bound2)]
    
    ret_robust.plot.scatter(x='v', y='y', c='DarkBlue', s=2)
    plt.title('signal return v.s. price return - central 90%')
    plt.xlabel('signal return' )
    plt.ylabel('price return')
    
    ret_outlier.plot.scatter(x='v', y='y', c='DarkBlue', s=2)
    plt.title('signal return v.s. price return - outliers')
    plt.xlabel('signal return' )
    plt.ylabel('price return')
    
    bad_data.plot.scatter(x='v', y='y', c='DarkBlue', s=2)
    plt.title('signal return v.s. price return - bad data')
    plt.xlabel('signal return' )
    plt.ylabel('price return')
    
    #lag corr plot
    lag2mean_corr = {}
    for lag in range(0, 60):
        lag2mean_corr[lag] = ret['v'].to_frame().rollingcorr(window=10, col=ret['y'].shift(lag)).mean().values[0]
    pd.DataFrame(lag2mean_corr, index=['mean_corr']).T.plot()
    plt.title('avg rank corr v.s. signal lag')
    plt.xlabel('# lag 10min interval' )
    plt.ylabel('average rank corr')