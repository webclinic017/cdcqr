import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from croqr.common.config import LOCAL_DATA_DIR, LOCAL_FIGURE_DIR, LOCAL_LOG_DIR
from croqr.common.utils import timeit, setup_custom_logger
from croqr.data.cq.config import CryptoQuantData
from croqr.data.cq.utils import align_feature_df, get_feature_df_corr_with_ret
from datetime import datetime
import argparse

if __name__ == '__main__':
    
    my_parser = argparse.ArgumentParser(description='calcualte time series correlation between returns and features')
    my_parser.add_argument('--test', metavar='test', type=str, help='test mode', dest="test", default=False)
    my_parser.add_argument('--test_size', metavar='test_size', type=int, help='test mode truncation size', dest="test_size", default=5000)
    args = my_parser.parse_args()
    test_mode = args.test
    test_size = args.test_size
        
    logformat = "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"
    datefmt = "%m-%d %H:%M"
    log_file_name = 'cq_corr_{}.log'.format(datetime.now().strftime('%Y%m%d%H%M%S'))

    logging.basicConfig(filename=os.path.join(LOCAL_LOG_DIR, log_file_name), level=logging.INFO, filemode="w",
                        format=logformat, datefmt=datefmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))

    logger = logging.getLogger("app")
    logger.addHandler(stream_handler)


    if test_mode:
        logger.info('test mode is on')

    file_path = os.path.join(LOCAL_DATA_DIR, 'cq2.pkl')
    logger.info('loading {}'.format(file_path))
    data_dict = pd.read_pickle(file_path)

    all_feature_list = list(data_dict.keys())
    all_feature_list.remove('btc-all_exchange-market-data-price-usd')
    existing_table_anmes = existing_files = [x.split('.')[0] for x in os.listdir(LOCAL_FIGURE_DIR)]
    all_feature_list = [x for x in all_feature_list if x not in existing_table_anmes]
    btc_close = data_dict['btc-all_exchange-market-data-price-usd']['price_usd_close'][::-1]
    ret_df = btc_close.pct_change().fillna(0)

    if test_mode:
        ret_df = ret_df.head(test_size)


    feature_exclusion_list = ['blockheight', 'datetime']

    for feature_table_name in all_feature_list:
        try:
            logger.info('Working on feature table: {}'.format(feature_table_name))
            feature_table = data_dict[feature_table_name]
            feature_list = [x for x in feature_table.columns if x not in feature_exclusion_list]
            logger.info('Feature table columns: {}'.format(feature_list))
            if test_mode:
                refeature_table_df = feature_table.head(test_size*50)

            aligned_feature_df = align_feature_df(feature_table, feature_list, ret_df)
            get_feature_df_corr_with_ret(aligned_feature_df, feature_table_name, ret_df)

        except Exception:
            pass