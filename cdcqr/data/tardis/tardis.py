from tardis_dev import datasets
from cdcqr.common.config import LOCAL_DATA_DIR
import os

#data_path= r'C:\Users\Hassan\Documents\data/'
API_KEY="TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4"
TARDIS_DOWNLOAD_DIR = os.path.join(LOCAL_DATA_DIR, 'tardis')

#TARDIS_DOWNLOAD_DIR= r"C:\Users\Hassan\Documents\data\Tardis"

datasets.download(exchange="deribit",
                  #data_types=["incremental_book_L2", "trades", "quotes", "derivative_ticker", "book_snapshot_25",
                  #            "liquidations"],
                  data_types=['quotes' ],
                  
                  from_date="2021-06-04",
                  to_date="2021-06-05",
                  symbols=['ETH-27AUG21-2000-P'], #, "ETH-PERPETUAL", 'OPTIONS'],#  symbols=["BTC-9JUN20-9875-P"],
                  #symbols=["BTC-9JUNE20-9875-P"], #, "ETH-PERPETUAL", 'OPTIONS'],#  symbols=["BTC-9JUN20-9875-P"],
                  api_key=API_KEY,
                  download_dir=TARDIS_DOWNLOAD_DIR)
if __name__ == '__main__':
    pass