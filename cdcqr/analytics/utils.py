import matplotlib.pyplot as plt
from cdcqr.ct.utils import plot2, rollingcorr
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad



pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def rollingcorr(df, window, col=None, min_periods=None, method='spearman'):
    if method =='pearson':
        return df.rolling(window).corr(col)
    elif method=='spearman':
        return rolling_rank_corr(df, window, y = col, min_periods=1)
    else:
        raise ValueError(f"method {method} not implemented")

pd.core.frame.DataFrame.rollingcorr=rollingcorr


def rolling_spearman_quick(seqa, seqb, window):
    seqa =seqa.values
    seqb =seqb.values
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)


def rolling_rank_corr(df, window, y = None, min_periods=1):
    if y is None:
        n, p = df.shape
        x = df.values

        # create rolling 3d array
        mat_3d = as_strided(x, shape=(n-window+1, window, p), strides=(x.strides[0], x.strides[0], x.strides[1]))

        # apply spearman corr
        res = np.array([libalgos.nancorr_spearman(x) for x in mat_3d])

        # produce missing array
        a = np.empty((window-1, p, p,))
        a[:] = np.nan

        # concat
        res2 = np.concatenate((a, res))

        # create index
        t = [[i]*p for i in df.index]
        flat_list1 = [item for sublist in t for item in sublist]
        t2 = [df.columns]*n
        flat_list2 = [item for sublist in t2 for item in sublist]
        arrays = [flat_list1, flat_list2]
        midx = pd.MultiIndex.from_arrays(arrays)

        return pd.DataFrame(res2.reshape(n*p, p), index=midx, columns=df.columns)
    else:
        ret_dict = {}
        for col in df.columns:
            ret_dict[col] = rolling_spearman_quick(df[col], y, window=window)
        return pd.DataFrame(ret_dict, index=df.index)


def ts_eda(dff0, x, xn, y, lbw = 20):
    dff = dff0.copy()
    # double y plot
    cols = dff.columns

    dff['xn_pctrank'] = dff[xn].rolling(lbw).apply(pctrank)
    dff['xn_zscore'] = zscore(dff[xn], lbw)

    dff.plot2(x, y)
    dff.plot2(xn, y)
    dff.plot2('xn_pctrank', y)
    dff.plot2('xn_zscore', y)

    n_f = 3
    n_lag = 3
    n_group = 3
    fig = plt.figure(num=1, figsize=(6 * n_f, 4 * n_lag*n_group))
    dff['yr'] = dff[y].pct_change()

    for i, feature in enumerate([xn,'xn_pctrank','xn_zscore']):
        for j, lag in enumerate([0,1,2]):
            dff['f_lagged'] = dff[feature].shift(lag)
            lower_bound1 = dff[xn].quantile(0.05)
            upper_bound1 = dff[xn].quantile(0.95)
            lower_bound2 = dff[xn].quantile(0.01)
            upper_bound2 = dff[xn].quantile(0.99)

            mid90 = dff[(dff[xn] <= upper_bound1) & (dff[xn] >= lower_bound1)]
            outlier8 = dff[((dff[xn] > upper_bound1) & (dff[xn] < upper_bound2)) | (
                        (dff[xn] < lower_bound1) & (dff[xn] > lower_bound2))]
            outlier2 = dff[(dff[xn] >= upper_bound2) | (dff[xn] <= lower_bound2)]
            group_dict={}
            group_dict[0] = 'mid90'
            group_dict[1] = 'outlier8'
            group_dict[2] = 'outlier2'
            for k, data in enumerate([mid90, outlier8, outlier2]):
                ax = fig.add_subplot(n_lag * n_group, n_f, i * n_lag * n_group + j * n_group + k + 1)
                #data.plot.scatter(x='f_lagged', y='yr', c='DarkBlue', s=2, ax=ax)

                dflr = data[[feature,'yr']].dropna()

                x1 = '{}_lag{}_{}'.format(feature, lag, group_dict[k])
                y1 = 'yr'
                dflr.columns = [x1, y1]
                lr=LinearRegression(fit_intercept=False).fit(dflr[[x1]],dflr[y1])
                lri=LinearRegression(fit_intercept=True).fit(dflr[[x1]],dflr[y1])
                sns.scatterplot(x=x1,y=y1,data=dflr, ax = ax)
                ax.plot(dflr[x1],lr.predict(dflr[[x1]]),color='g')
                ax.plot(dflr[x1],lri.predict(dflr[[x1]]),color='r')
                plt.title(f"r {lr.coef_[0]:.3f}x R2 {r2_score(dflr[[y1]],lr.predict(dflr[[x1]])):.3f} g {lri.coef_[0]:.3f}x+{lri.intercept_:.3f} R2 {r2_score(dflr[[y1]],lri.predict(dflr[[x1]])):.3f}")


                #plt.title('{} lag {} {} v.s. {}'.format(feature, lag, group_dict[k],'{} return'.format(y)), )

        fig.tight_layout()


    if True:
        # return scatter plots
        f = x 
        u = y
        dff[f] = dff[f].rolling(20, min_periods=1).mean()
        ret = dff.pct_change()
        ret = ret[ret[f].notna()]
        ret = ret[ret[f] != 0]
        

        #lag corr plot
        lag2mean_corr = {}
        for lag in range(0, 60):
            lag2mean_corr[lag] = ret[f].to_frame().rollingcorr(window=10, col=ret[u].shift(lag)).mean().values[0]
        pd.DataFrame(lag2mean_corr, index=['mean_corr']).T.plot()
        plt.title('avg rank corr v.s. signal lag')
        plt.xlabel('# lag 10min interval')
        plt.ylabel('average rank corr')
        fig.tight_layout()
        return ret
