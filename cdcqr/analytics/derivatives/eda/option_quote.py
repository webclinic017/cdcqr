import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from cdcqr.common.config import TARDIS_API_KEY, LOCAL_DATA_DIR, LOCAL_FIGURE_DIR
from tardis_dev import datasets
import os
from cdcqr.data.deribit.data_utils import DeribitUtils
from plotly.subplots import make_subplots
import plotly.graph_objects as go


TARDIS_DOWNLOAD_DIR = os.path.join(LOCAL_DATA_DIR, 'tardis')

def load_option_quote_and_perp_data(date, coin):

    # load - option quote data
    exchange = 'deribit'
    data_type = 'quotes'
    symbols = ['OPTIONS', '{}-PERPETUAL'.format(coin)]
    date_ = date.strftime('%Y-%m-%d')
    datasets.download(exchange="deribit",
        data_types=[data_type],
        from_date=date_,
        to_date=date_,
        symbols=symbols,
        api_key=TARDIS_API_KEY,
        download_dir=TARDIS_DOWNLOAD_DIR)

    fname = '{}_{}_{}_{}.csv.gz'.format(exchange, data_type, date_, symbols[0])
    print('download and read data at {}'.format(os.path.join(TARDIS_DOWNLOAD_DIR, fname)))
    opt_quote = pd.read_csv(os.path.join(TARDIS_DOWNLOAD_DIR, fname))
    
    fname = '{}_{}_{}_{}.csv.gz'.format(exchange, data_type, date_, symbols[1])
    print('download and read data at {}'.format(os.path.join(TARDIS_DOWNLOAD_DIR, fname)))
    PERP_quote = pd.read_csv(os.path.join(TARDIS_DOWNLOAD_DIR, fname))
    return opt_quote, PERP_quote

def load_optchain_data(date, maturity_date, freq):
    start_date = date
    end_date = date
    file_name = 'optchain_{}_{}_{}_{}.pkl'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), maturity_date.strftime('%Y%m%d'), freq)
    optchain = pd.read_pickle(os.path.join(LOCAL_DATA_DIR, file_name))
    return optchain



def deribit_option_quote_plot(date, maturity_date, freq='1Min', coin='BTC'):
    """
    produce three interactive scatter plots based on data items below:
        deribit option quote -> get best bid/ask price/quantity 
        deribt PERP data
        deribit option chain data -> get bid.ask IVs and option underlying price

    three scatter plots are respectively for 
        ask quotes amount across put/call and all strikes 
        bid quotes amount across put/call and all strikes 
        ask+bid quote amount across put/call and all strikes 
    """
    opt_quote, PERP_quote = load_option_quote_and_perp_data(date, coin)
    optchain = load_optchain_data(date, maturity_date, freq)

    ### data processing - all
    uni_index = pd.date_range(start=date, end=date+timedelta(days=1), freq='1Min')[1:]

    #### data processing - PERP quote
    PERP_quote['dt'] = pd.to_datetime(PERP_quote['timestamp'], unit='us')
    PERP_quote['PERP_mid_price'] = (PERP_quote['ask_price']+PERP_quote['bid_price'])/2
    PERP_quote_1 = PERP_quote.set_index('dt')['PERP_mid_price']
    PERP_quote_2 = PERP_quote_1[~PERP_quote_1.index.duplicated(keep='last')]
    PERP_quote_3 = PERP_quote_2.reindex(uni_index, method='ffill')

    #### data processing - opt chain
    optchain['mid_vol'] = 0.5*(optchain['aiv'] + optchain['biv'])
    optchain_reduced = optchain[['symbol','strike','underly','s','aiv','biv']]
    optchain_reduced = optchain_reduced[optchain_reduced['symbol'].str.contains('{}-'.format(coin))]
    optchain_reduced = optchain_reduced.pipe(DeribitUtils.parse_optSymbol_col)

    # 1. get underlying price
    optchain_reduced_1 = optchain_reduced['s']
    optchain_reduced_2 = optchain_reduced_1[~optchain_reduced_1.index.duplicated(keep='last')]
    optchain_reduced_2 = optchain_reduced_2.sort_index()
    optchain_reduced_3 = optchain_reduced_2.reindex(uni_index, method='ffill').to_frame()
    optchain_reduced_3.columns = ['underlying_price']
    df_underlying = optchain_reduced_3.join(PERP_quote_3).reset_index()

    # 2. get ask iv per strike, per type
    optchain_reduced_1a = optchain_reduced[['aiv', 'biv', 'strike', 'type']]
    optchain_reduced_1a = optchain_reduced_1a.reset_index().rename(columns={'tm':'index'})
    # remove duplicates in opt chain
    optchain_reduced_1a = optchain_reduced_1a[~optchain_reduced_1a.set_index(['type','strike','index']).index.duplicated(keep='last')]

    #### data processing - option quote
    opt_quote_parsed = opt_quote.pipe(DeribitUtils.parse_optSymbol_col)
    opt_quote_parsed['timestamp_dt'] = pd.to_datetime(opt_quote_parsed['timestamp'], unit='us')

    # only look at BTC
    opt_quote_parsed = opt_quote_parsed.query('instrument=="BTC"')
    opt_quote_parsed['exp_date'] = opt_quote_parsed['expire'].dt.date

    # select relevant expire dates
    opt_quote_parsed_expire_i_c = opt_quote_parsed.query('exp_date==@maturity_date & type =="C"')
    opt_quote_parsed_expire_i_p = opt_quote_parsed.query('exp_date==@maturity_date & type =="P"')

    #### reorganzie the data along diffrerent strikes
    strike_list = list(sorted(opt_quote_parsed_expire_i_c['strike'].unique()))
    opt_quote_parsed_expire_i_c_reduced = opt_quote_parsed_expire_i_c.drop(labels=['exchange','symbol','timestamp','local_timestamp','type','expire','t2m'], axis=1, errors='ignore')
    opt_quote_parsed_expire_i_p_reduced = opt_quote_parsed_expire_i_p.drop(labels=['exchange','symbol','timestamp','local_timestamp','type','expire','t2m'], axis=1, errors='ignore')

    strike2df = {}
    for strike in strike_list:
        opt_quote_parsed_expire_i_c_reduced_k_1 = opt_quote_parsed_expire_i_c_reduced.query('strike==@strike').set_index('timestamp_dt')
        opt_quote_parsed_expire_i_c_reduced_k_1 = opt_quote_parsed_expire_i_c_reduced_k_1[~opt_quote_parsed_expire_i_c_reduced_k_1.index.duplicated(keep='last')]
        opt_quote_parsed_expire_i_c_reduced_k_1 = opt_quote_parsed_expire_i_c_reduced_k_1.reindex(uni_index, method='ffill')
        strike2df[strike] = opt_quote_parsed_expire_i_c_reduced_k_1
    df_combined = pd.concat(strike2df.values()).reset_index()
    df_combined['ask_amount'] = df_combined['ask_amount'].fillna(0)
    df_combined['bid_amount'] = df_combined['bid_amount'].fillna(0)
    df_combined_c = df_combined.copy()

    strike2df = {}
    for strike in opt_quote_parsed_expire_i_p_reduced['strike'].unique():

        opt_quote_parsed_expire_i_p_reduced_k_1 = opt_quote_parsed_expire_i_p_reduced.query('strike==@strike').set_index('timestamp_dt')
        opt_quote_parsed_expire_i_p_reduced_k_1 = opt_quote_parsed_expire_i_p_reduced_k_1[~opt_quote_parsed_expire_i_p_reduced_k_1.index.duplicated(keep='last')]
        opt_quote_parsed_expire_i_p_reduced_k_1_1min = opt_quote_parsed_expire_i_p_reduced_k_1.reindex(uni_index, method='ffill')
        strike2df[strike] = opt_quote_parsed_expire_i_p_reduced_k_1_1min
    df_combined = pd.concat(strike2df.values()).reset_index()
    df_combined['ask_amount'] = df_combined['ask_amount'].fillna(0)
    df_combined['bid_amount'] = df_combined['bid_amount'].fillna(0)
    df_combined_p = df_combined.copy()

    df_combined_c['type'] = 'C'
    df_combined_p['type'] = 'P'
    df_combined = df_combined_c.append(df_combined_p)

    # shift positions to seprate put and call options
    df_combined['strike_shifted'] = df_combined['strike'] + 100*((df_combined['type']=='C').astype(int)-0.5)*2

    #### data processing - all
    df_combined_ext = pd.merge(left=df_combined, right = optchain_reduced_1a, on=['index','strike','type'], how='left')
    df_combined_ext['mvol'] = 0.5*(df_combined_ext['aiv']+df_combined_ext['biv'])
    df_combined_ext['total_amount'] = df_combined_ext['bid_amount'] + df_combined_ext['ask_amount']

    ### make plot
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = px.scatter(df_combined_ext, x="index", y="strike_shifted", size="ask_amount",  hover_data=['strike','ask_price','ask_amount','type', 'biv'], color="biv", title='',width=1500, height=1200)
    fig1b = px.scatter(df_combined_ext, x="index", y="strike_shifted", size="bid_amount",  hover_data=['strike','bid_price','bid_amount','type', 'biv'], color="biv", title='',width=1500, height=1200)
    fig1c = px.scatter(df_combined_ext, x="index", y="strike_shifted", size="total_amount", hover_data=['strike','ask_price','ask_amount','bid_price','bid_amount','type', 'aiv','biv','mvol'], color="mvol", title='',width=1500, height=1200)

    fig2 = px.line(df_underlying, x='index', y=['underlying_price', 'PERP_mid_price'], width=1500, height=1200)

    fig3a = go.Figure(data=fig1.data + fig2.data,)
    fig3a.update_layout(
        autosize=False,
        width=1500,
        height=1200,
        title="{} ask price and amount v.s. underlying & PERP date:{}, k:{}, expire:{}".format(coin, date, strike, maturity_date))
    fig3a.show()
    fig3a.write_html(os.path.join(LOCAL_FIGURE_DIR,'option_book',date.strftime('%Y%m'), 
                             "{}_options_{}_{}_ask.html".format(coin, date.strftime('%Y%m%d'), maturity_date.strftime('%Y%m%d'))))

    fig3b = go.Figure(data=fig1b.data + fig2.data,)
    fig3b.update_layout(
        autosize=False,
        width=1500,
        height=1200,
        title="{} bid price and amount v.s. underlying & PERP date:{}, k:{}, expire:{}".format(coin, date, strike, maturity_date))
    fig3b.show()
    fig3b.write_html(os.path.join(LOCAL_FIGURE_DIR,'option_book', date.strftime('%Y%m'),
                                "{}_options_{}_{}_bid.html".format(coin, date.strftime('%Y%m%d'), maturity_date.strftime('%Y%m%d'))))


    fig3c = go.Figure(data=fig1c.data + fig2.data,)
    fig3c.update_layout(
        autosize=False,
        width=1500,
        height=1200,
        title="{} bid+ask amount v.s. underlying & PERP date:{}, k:{}, expire:{}".format(coin, date, strike, maturity_date))
    fig3c.show()
    fig3c.write_html(os.path.join(LOCAL_FIGURE_DIR, 'option_book', date.strftime('%Y%m'),
                                "{}_options_{}_{}_all_quotes.html".format(coin, date.strftime('%Y%m%d'), maturity_date.strftime('%Y%m%d'))))


if __name__ == '__main__':
    date_ = datetime(2021,10,29).date()
    maturity_date = datetime(2022,6,24).date()
    deribit_option_quote_plot(date_, maturity_date)