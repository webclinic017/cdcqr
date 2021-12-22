import logging
import multiprocessing
import os
import pandas as pd
import sys
import time
from collections import OrderedDict
from cdcqr.common.config import LOCAL_DATA_DIR
from functools import wraps
import re
from pandas_flavor import register_dataframe_method


def timeit(method):
    """
    Decorator used to time the execution time of the downstream function.
    :param method: downstream function
    """

    @wraps(method)
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        if end_time - start_time > 3:
            print('%r  %2.2f sec' % (method.__name__, end_time - start_time))
        return result

    return timed


@timeit
def parallel_jobs(func2apply, domain_list,
                  message='processing individual tasks', num_process=None):
    """
    use multiprocessing module to parallel job
    return a dictionary containing key and corresponding results
    """
    ret_dict = OrderedDict()
    with multiprocessing.Pool(num_process) as pool:
        for idx, ret in enumerate(
                pool.imap_unordered(func2apply, domain_list)):
            ret_dict[domain_list[idx]] = ret
            sys.stderr.write('\r{0} {1:%}'.format(message,
                                                  idx / len(domain_list)))

    return ret_dict


def print_time_from_t0(start_time):
    end_time = time.time()
    print('%2.2f sec' % (end_time - start_time))


def setup_custom_logger(abs_file, log_level=logging.DEBUG):
    """
    Sets up the custom logger with logging formats applied.
    :param abs_file: absolute log file path
    :param log_level: logging level
    :return: logger instance
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # create/open log file
    handler = logging.FileHandler(abs_file)
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(abs_file)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


@register_dataframe_method
def resample_pv(df1, freq='1D'):
    def agg_f(cols):
        ret = {}
        for col in cols:
            if col in ['o','h','l','c']:
                ret[col] = 'last'
            elif col=='v':
                ret[col] = 'sum'
        return ret
    return df1.resample(freq).agg(agg_f(df1.columns))



@register_dataframe_method
def addbb(df, lags, cols=None, inplace=False, methodma='ewm', methodstd='ewm', nstdh=2, nstdl=2, retfnames=False,
          dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            ma = df[col].addma(lag, method=methodma)
            std = df[col].addstd(lag, method=methodstd)
            fname = 'bbh.' + col
            df[fname] = ma + nstdh * std
            fnames.append(fname)
            fname = 'bbl.' + col
            df[fname] = ma - nstdl * std

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df

    
@register_dataframe_method
def save(df, name='no_name', file_format='pickle'):
    file_path = os.path.join(LOCAL_DATA_DIR, '{}.{}'.format(name, file_format))
    if file_format=='pickle':
        df.to_pickle(file_path)
    elif file_format=='csv':
        df.to_csv(file_path)
    print('saved df to {}'.format(file_path))


def load_df(name, file_format='pickle'):

    if file_format=='pickle':
        try:
            file_path = os.path.join(LOCAL_DATA_DIR, '{}.{}'.format(name, file_format))
            df = pd.read_pickle(file_path)
            return df
        except:
            pass
        
        try:
            file_path = os.path.join(LOCAL_DATA_DIR, '{}.{}'.format(name, 'pkl'))
            df = pd.read_pickle(file_path)
            return df
        except:
            pass
            
    elif file_format=='csv':
        df = pd.read_csv(file_path)
        return df



def camel_case2snake_case(camel_case):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case).lower()