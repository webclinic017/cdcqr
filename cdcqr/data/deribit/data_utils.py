import io
import os
import urllib.request

import pandas as pd
import numpy as np
import requests

from cdcqr.common.config import DATA_CACHE_DIR
from cdcqr.common.utils import timeit
from datetime import timedelta
from cdcqr.data.dataloader import data_loader
import warnings
warnings.filterwarnings("ignore")


@timeit
def url_csv_zip_file_reader(url, cache_dir=DATA_CACHE_DIR, use_cache=True):
    """
    Returns the DataFrame from url zipped file.

        Parameters:
                url (str): zipped file url. e.g. data.csv.gz
                cache_dir (str): path to store data cached
                use_cache (bool): use cache or not
        Returns:
                df (DataFrame): DataFrame of the data content
    """
    file_name = url.split('/')[-1]
    file_path = os.path.join(cache_dir, file_name)
    try:
        if use_cache:
            if os.path.isfile(file_path):
                print('loading cached', file_name)
                if os.path.getsize(file_path) > 500e6:
                    print('file size > 500MB, load top 5000 rows')
                    return pd.read_csv(file_path, compression="gzip", index_col=0, quotechar='"', nrows=5000)
                else:
                    return pd.read_csv(file_path, compression="gzip", index_col=0, quotechar='"')

            else:
                print('downloading and caching', url)
                opener = urllib.request.URLopener()
                # specify headers so that the website will allow Python to read its data
                opener.addheader('User-Agent', 'whatever')
                _ = opener.retrieve(url, file_path)
                if os.path.getsize(file_path) > 500e6:
                    print('file size > 500MB, load top 5000 rows')
                    return pd.read_csv(file_path, compression="gzip", index_col=0, quotechar='"', nrows=5000)
                else:
                    return pd.read_csv(file_path, compression="gzip", index_col=0, quotechar='"')
        else:
            # specify headers so that the website will allow Python to read its data
            user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
            headers = {'User-Agent': user_agent, }

            response = requests.get(url, headers)
            content = response.content
            df = pd.read_csv(
                io.BytesIO(content), sep=",", compression="gzip", index_col=0, quotechar='"',
            )
            df.to_pickle(file_path)
            return df
    except:
        print(url, 'not exists!')
        pass

    

class DeribitUtils:
    def __init__(self):
        pass

    """
    Returns the parsed conent of an option symbol string

        Parameters:
                symbol (str): e.g. 'BTC-31DEC21-36000-P'
        Returns:
                instrument (str): BTC
                expire_date (datetime): datetime(2021, 12, 31)
                strike (float): 36000
                option_type (string): 'P' 
    """
    def optSymbo2instrument(symbol):
        return symbol.split('-')[0]

    def optSymbo2expire(symbol):
        return symbol.split('-')[1]

    def optSymbo2strike(symbol):
        return symbol.split('-')[2]

    def optSymbo2type(symbol):
        return symbol.split('-')[3]

    @timeit
    def parse_optSymbol_col(df0):
        df = df0.copy()
        df['instrument'] = df['symbol'].apply(DeribitUtils.optSymbo2instrument)

        df['expire'] = df['symbol'].apply(DeribitUtils.optSymbo2expire)
        df['expire'] = pd.to_datetime(df['expire'])

        df['strike'] = df['symbol'].apply(DeribitUtils.optSymbo2strike).astype(float)

        df['type'] = df['symbol'].apply(DeribitUtils.optSymbo2type)
    
        return df

    @timeit
    def parse_time_col(df0):
        df = df0.copy()
        df['timestamp_dt'] =pd.to_datetime(df['timestamp'], unit='us')
        if 'expire' in df.columns:
            df['expire'] = df['expire'] + timedelta(hours=8)
            df['t2m'] = (df['expire'] - df['timestamp_dt'].dt.floor('H')).dt.total_seconds()/3600
        return df

        
    
    @timeit
    def parse_futureSymbol_col(df0):
        df = df0.copy()
        df['instrument'] = df['symbol'].apply(DeribitUtils.futureSymbol2instrument)
        return df


    def futureSymbol2instrument(symbol):
        return symbol.split('-')[0]

    
    
def get_order_book_dynamics(contract='BTC-8OCT21-48000-C', date ="2021-10-01" , symbol='options', exchange='deribit'):
    display('BTC-8OCT21-48000-C', "2021-10-01", 'deribit')
    if symbol=='options':
        df_opt_quote = data_loader(exchange, date,"quotes",symbol).pipe(DeribitUtils.parse_optSymbol_col).pipe(DeribitUtils.parse_time_col)
        df_opt_trade = data_loader(exchange, date,"trades",symbol).pipe(DeribitUtils.parse_optSymbol_col).pipe(DeribitUtils.parse_time_col)
    else:
        df_opt_quote = data_loader(exchange, date,"quotes",symbol).pipe(DeribitUtils.parse_time_col)
        df_opt_trade = data_loader(exchange, date,"trades",symbol).pipe(DeribitUtils.parse_time_col)
    ATM_btc_c = contract
    
    # processing trade info
    df_opt_trade_i = df_opt_trade.query('symbol==@ATM_btc_c')
    df_opt_trade_i['flow_sgn'] = (df_opt_trade_i['side']=='buy').astype(int)
    df_opt_trade_i['flow_sgn'] = (df_opt_trade_i['flow_sgn']-0.5)*2
    df_opt_trade_i['flow'] = df_opt_trade_i['flow_sgn']*df_opt_trade_i['amount']
    df_opt_trade_i_no_dup = df_opt_trade_i.groupby('timestamp_dt')['flow'].sum().reset_index()
    df_opt_trade_i_no_dup = df_opt_trade_i_no_dup.rename(columns={'timestamp_dt':'trade_time'})
    
    
    # processing quote info
    df_opt_quote_i = df_opt_quote.query('symbol==@ATM_btc_c')
    df_opt_quote_i_no_dup = df_opt_quote_i[~df_opt_quote_i['timestamp_dt'].duplicated(keep='last')]
    
    # align timestamps
    B = df_opt_quote_i_no_dup['timestamp_dt'].values
    A = df_opt_trade_i_no_dup['trade_time'].values
    res = B[np.searchsorted(B, A)]
    df_opt_trade_i_no_dup['timestamp_dt'] = res
    
    df_opt_trade_i_no_dup2 = df_opt_trade_i_no_dup.groupby('timestamp_dt')['flow'].sum().reset_index()
    df_opt_trade_i_no_dup2['has_trade'] = True
    
    # combine
    df_combined = pd.merge(left=df_opt_quote_i_no_dup, right=df_opt_trade_i_no_dup2, on=['timestamp_dt'], how='left')
    
    # adding features
    df_combined['flow'] = df_combined['flow'].fillna(0)
    df_combined['trade_neighbour'] = df_combined['has_trade']
    df_combined['trade_neighbour'] =df_combined['has_trade'].ffill(limit=1).bfill(limit=1).fillna(False)
    
    df_combined['mid_price'] = (df_combined['ask_price'] + df_combined['bid_price'])/2
    df_combined['wgt_mid_price'] = (df_combined['ask_price']*df_combined['ask_amount'] + df_combined['bid_price']*df_combined['bid_amount'])/(df_combined['ask_amount']+df_combined['bid_amount'])
    df_combined['midp_chg'] = df_combined['mid_price'].pct_change()
    df_combined['wgtmidp_chg'] = df_combined['wgt_mid_price'].pct_change()
    
    num_trades = df_combined['has_trade'].sum()
    num_midp_chg = (df_combined['midp_chg']!=0).sum()
    num_wgtmidp_chg = (df_combined['wgtmidp_chg']!=0).sum()
    return {'num_trades':num_trades,'num_midp_chg':num_midp_chg,'num_wgtmidp_chg':num_wgtmidp_chg}, df_combined