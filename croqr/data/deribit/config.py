import dataclasses


@dataclasses.dataclass
class DeribitSampleDataUrl:
    date_range = '2020-07-01_2020-07-31'
    file_format = 'csv.gz'
    data_source = 'Deribit'
    asset_class_list = ['PERPETUALS', 'FUTURES', 'OPTIONS']
    data_type_list = ['trades', 'book_snapshot_50', 'incremental_book_L2', 'quotes', 'options_chain',
                      'derivative_ticker']


def get_deribit_sample_data_url(asset_class, data_type):
    return "https://csv.tardis.dev/samples/{}_{}_{}_{}.{}".format(DeribitSampleDataUrl.data_source,
                                                                  asset_class,
                                                                  data_type,
                                                                  DeribitSampleDataUrl.date_range,
                                                                  DeribitSampleDataUrl.file_format)
