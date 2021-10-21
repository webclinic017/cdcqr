import matplotlib.pyplot as plt
import os
import pandas as pd
from cdcqr.common.config import LOCAL_DATA_DIR, LOCAL_FIGURE_DIR
from cdcqr.common.utils import timeit
from cdcqr.data.cq.config import CryptoQuantData
from datetime import datetime
from IPython.display import display


@timeit
def align_feature_df(raw_feature_df, feature_list, ret_df):
    # reverse time index
    feature_df = raw_feature_df[::-1].reset_index()
    print('-------------')
    print(raw_feature_df.shape)
    print(feature_df.head())
    # get signal_time
    feature_df['signal_time'] = feature_df['datetime'].apply(lambda x: x.ceil('min'))

    # select relevant columns
    feature_df = feature_df[feature_list + ['signal_time']].set_index('signal_time')

    # drop duplicated index
    feature_df = feature_df[~feature_df.index.duplicated(keep='first')]

    # align to ret dataframe
    aligned_feature_df = feature_df.reindex(index=ret_df.index).ffill()

    return aligned_feature_df


@timeit
def get_feature_df_corr_with_ret(df_features, feature_table_name, ret_df, look_back_window=60*24):
    feature_list = list(df_features.columns)
    df_features_chg = df_features.pct_change().fillna(0)
    n = len(feature_list)

    plt.tight_layout(pad=0.5, w_pad=2.5, h_pad=2.0)
    fig = plt.figure(figsize=(6, 4 * n))

    for i, feature in enumerate(feature_list):
        ax1 = fig.add_subplot(n, 1, i + 1)
        ax1.grid(True, axis='x')
        f_chg = df_features_chg[feature]
        f_chg.rolling(look_back_window).corr(ret_df).rolling(look_back_window * 60).mean().plot(
            ax=ax1)
        f_chg.shift(1).rolling(look_back_window).corr(ret_df).rolling(
            look_back_window * 60).mean().plot(ax=ax1)
        f_chg.shift(5).rolling(look_back_window).corr(ret_df).rolling(
            look_back_window * 60).mean().plot(ax=ax1)
        f_chg.shift(30).rolling(look_back_window).corr(ret_df).rolling(
            look_back_window * 60).mean().plot(ax=ax1)
        ax1.hlines(y=0, xmin=f_chg.index.min(), xmax=f_chg.index.max(), linewidth=2, color='r', label=['0'], Linestyles='dotted')
        ax1.legend(['lag=0', 'lag=1', 'lag=5', 'lag=30', '0'])
        ax1.title.set_text(feature)
        ax1.get_xaxis().set_visible(False)

    fig.savefig(os.path.join(LOCAL_FIGURE_DIR, '{}.png'.format(feature_table_name)))
