import os
from cdcqr.common.config import LOCAL_DATA_DIR
from tardis_dev import datasets


TARDIS_DOWNLOAD_DIR = os.path.join(LOCAL_DATA_DIR, 'tardis')

API_KEY = 'TD.sDyJS7YZ6oPWSgy2.-vZySO46Lv8avKO.ixQvOq9xdhxqnzC.p1rlPahcqt4F3pp.uORrUOeq0hqYOhV.w6s4'

datasets.download(exchange="deribit",
                  # data_types=["incremental_book_L2", "trades", "quotes", "derivative_ticker", "book_snapshot_25",
                  #            "liquidations"],
                  data_types=["trades", 'quotes'],

                  from_date="2021-10-29",
                  to_date="2021-10-29",
                  symbols=["BTC-PERPETUAL", "ETH-PERPETUAL", 'OPTIONS'],
                  # symbols=['FUTURES'],
                  api_key=API_KEY,
                  download_dir=TARDIS_DOWNLOAD_DIR)

if __name__ == '__main__':
    pass
