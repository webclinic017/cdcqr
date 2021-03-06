from cdcqr.ct.utils import dt2ts
from cdcqr.common.utils import timeit, camel_case2snake_case
from datetime import datetime as dt
import time
import requests
import pandas as pd


def glassnode(url='https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_mean',a='BTC',i='10m',s=dt(2021,8,1), Nretry=10,API_KEY=""):
    """
    i
    1 month (1month)
    1 week (1w)
    1 day (24h)
    1 hour (1h)
    10 minutes (10m)
    
    """
    # print(url, a, i , s, u )
    timestamp_s=dt2ts(s,delta='1s')
    # print(timestamp_s)

    
    try:
        res = requests.get(url,
                           params={'a': a,'i':i,'s':timestamp_s,'api_key': API_KEY, }) #, 's':1629500000
        df = pd.read_json(res.text, convert_dates=['t']).set_index('t') 
    except Exception as e:
        print(e, res.text)
        if Nretry==0:
            raise e
        time.sleep(120)
        return glassnode(url=url,a=a,i=i,s=s,Nretry=Nretry-1,API_KEY=API_KEY)
    return df


@timeit
def get_glassnode_data(category, feature, underly, freq, start_time, api_key):
    url ='https://api.glassnode.com/v1/metrics/{}/{}'.format(category, feature)
    return glassnode(url, underly, freq, start_time, Nretry=10, API_KEY=api_key)


def quantile_info(dff):
    display(dff.head(5))
    q01 = dff.quantile(0.01).values[0]
    q02 = dff.quantile(0.02).values[0]
    q05 = dff.quantile(0.05).values[0]
    q95 = dff.quantile(0.95).values[0]
    q98 = dff.quantile(0.98).values[0]
    q99 = dff.quantile(0.99).values[0]
    print(q01, q02, q05, q95, q98, q99)
    return q01, q02, q05, q95, q98, q99


def url_parser(url):
    att_dict = {}
    for pair in url.split('&'):
        [k,v] = pair.split('=')
        att_dict[k] = v
        
    feature_camel_case = att_dict['m'].split('.')[1]
    att_dict['feature'] = camel_case2snake_case(feature_camel_case)
    att_dict['category'] = att_dict['m'].split('.')[0]
    return att_dict
