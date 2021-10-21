import io
import os
import urllib.request

import pandas as pd
import requests

from cdcqr.common.config import DATA_CACHE_DIR
from cdcqr.common.utils import timeit
from datetime import timedelta


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
