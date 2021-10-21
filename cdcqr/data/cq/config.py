import dataclasses


@dataclasses.dataclass
class CryptoQuantData:
    file_format = 'pkl'
    instrument_list = ['btc', 'stablecoin', 'erc20', 'eth']
    exchange_list = ['coinbase_pro',
                     'derivative_exchange',
                     'deribit',
                     'binance',
                     'all_exchange',
                     'spot_exchange']
    
    data_source = 'CryptoQuant'

    data_type_list = ['exchange-flows-inflow',
                         'exchange-flows-netflow',
                         'exchange-flows-outflow',
                         'exchange-flows-reserve',
                         'exchange-flows-supply',
                         'flow-indicator-exchange-shutdown-index',
                         'flow-indicator-exchange-whale-ratio',
                         'flow-indicator-fund-flow-ratio',
                         'flow-indicator-mpi',
                         'flow-indicator-stablecoins-ratio',
                         'fund-data-market-premium',
                         'inter-entity-flows-miner-to-exchange',
                         'market-data-funding-rates',
                         'market-data-liquidations',
                         'market-data-open-interest',
                         'market-data-price-usd',
                         'market-data-taker-buy-sell-stats',
                         'market-indicator-estimated-leverage-ratio',
                         'market-indicator-mvrv',
                         'market-indicator-stablecoin-supply-ratio',
                         'miner-flows-inflow',
                         'miner-flows-netflow',
                         'miner-flows-outflow',
                         'network-data-addresses-count',
                         'network-data-difficulty',
                         'network-data-fees-transaction',
                         'network-data-supply',
                         'network-indicator-nvm',
                         'network-indicator-nvt',
                         'network-indicator-nvt-golden-cross',
                         'network-indicator-puell-multiple',
                         'network-indicator-stock-to-flow']
