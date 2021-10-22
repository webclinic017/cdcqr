from scipy.stats.mstats import winsorize
import numpy as np

def winsorize_df(df0, cols2limits = {}):
    df = df0.copy()
    for col, win_params in cols2limits.items():
        df[col] = winsorize(df[col], limits = win_params, inclusive=(False, False))
    
    return df


def remove_outlier(df0, cols2limits = {}):
    df = df0.copy()
    df['keep'] = True
    for col, limits in cols2limits.items():
        lower_limit = limits[0]*100
        upper_limit = limits[1]*100

        df['keep'] = (df[col] >= np.percentile(df[col],lower_limit)).values & df['keep'].values
        df['keep'] = (df[col] <= np.percentile(df[col],upper_limit)).values & df['keep'].values

    df1 = df[df['keep']].drop(columns=['keep'], axis=1)
    return df1