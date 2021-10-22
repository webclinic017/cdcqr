import pandas as pd
import os
from cdcqr.common.config import LOCAL_DATA_DIR

TARDIS_DATA_PATH = os.path.join(LOCAL_DATA_DIR, 'tardis') 

def data_loader(exchange, date, data_type, symbol, condtition=[]):
    fname = '{}_{}_{}_{}.csv.gz'.format(exchange, data_type, date, symbol)
    df = pd.read_csv(os.path.join(TARDIS_DATA_PATH, fname))

    return df