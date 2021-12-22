import pandas as pd
import requests


class GlassnodeData:
    def __init__(self, api_key):
        self.api_key = api_key

    @staticmethod
    def get_meta_info(api_key):
        url = 'https://api.glassnode.com/v2/metrics/endpoints'
        res = requests.get(url,
                           params={'api_key': api_key})
        df = pd.read_json(res.text)
        df[[1, 2, 3, 'category', 'f']] = df['path'].str.split('/', 4, expand=True)
        df = df.drop(labels=[1, 2, 3, 'path', 'formats'], axis=1)
        return df
