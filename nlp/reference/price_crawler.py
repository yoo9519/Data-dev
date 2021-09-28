# -*- coding: utf-8 -*-


#####################
### import module ###
#####################
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
# from airflow.operators.python_operator import BranchPythonOperator
# from airflow.operators.bash_operator import BashOperator
from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime, timedelta
# from airflow.contrib.hooks.aws_hook import AwsHook
# from airflow.models import Variable
# from airflow.hooks.S3_hook import S3Hook
import gzip
import logging
import os
from io import StringIO
import boto3

import pandas as pd
import numpy as np
import time
import sys
import logging
import os
import shutil
import re



#####################
### Pandas Option ###
#####################
pd.options.display.float_format = '{:.5f}'.format



############################
### S3 bucket Parameters ###
############################
# s3_sensorflag_bucket = ''
# s3_conn_id = ''
# s3_bucket = ''
# jsonpaths_bucket = ''


#######################
### Bithumb Crawler ###
#######################
def bithumb_crawler():
    """
    Bithumb Crawl during 24H information
    """
    bithumb_url = 'https://api.bithumb.com/public/ticker/ALL_KRW'
    html = requests.get(bithumb_url)
    soup = BeautifulSoup(html.text, "html.parser")
    soup = soup.text.replace('{"status":"0000","data":{', '')

    currency = []
    timestamp_kst = []
    opening_price = []
    high_price = []
    low_price = []
    close_price = []
    acc_trade_value = []
    unit_traded = []

    for i in soup.split('},')[:-1]:
        currency.append(i.split(':')[0].replace('"', ''))
        timestamp_kst.append(datetime.utcfromtimestamp(
            int(soup.split('},')[-1].replace('"', '').replace('}}', '').replace('date:', '')) / 1000).strftime(
            '%Y-%m-%d'))
        opening_price.append(i.split(',')[0].split(':')[2].replace('"', ''))
        high_price.append(i.split(',')[3].split(':')[1].replace('"', ''))
        low_price.append(i.split(',')[2].split(':')[1].replace('"', ''))
        close_price.append(i.split(',')[1].split(':')[1].replace('"', ''))
        acc_trade_value.append(i.split(',')[8].split(':')[1].replace('"', ''))
        unit_traded.append(i.split(',')[7].split(':')[1].replace('"', ''))

    bithumb_df = pd.concat([pd.DataFrame(currency, columns=['currency'])
                               , pd.DataFrame(timestamp_kst, columns=['timestamp_kst'])
                               , pd.DataFrame(opening_price, columns=['opening_price'])
                               , pd.DataFrame(high_price, columns=['high_price'])
                               , pd.DataFrame(low_price, columns=['low_price'])
                               , pd.DataFrame(close_price, columns=['close_price'])
                               , pd.DataFrame(acc_trade_value, columns=['acc_trade_value'])
                               , pd.DataFrame(unit_traded, columns=['unit_traded'])]
                           , axis=1)

    bithumb_df['exchange'] = 'Bithumb'
    print("Bithumb Done!")

    return bithumb_df



#######################
### Coinone Crawler ###
#######################
def coinone_crawler():
    """
    Coinone Crawl during 24H information
    """
    coinone_url = 'https://api.coinone.co.kr/ticker?currency=all'
    html = requests.get(coinone_url)
    soup = BeautifulSoup(html.text, "html.parser")

    currency = []
    timestamp_kst = []
    opening_price = []
    high_price = []
    low_price = []
    close_price = []
    acc_trade_value = []
    unit_traded = []

    for i in soup.text.split(':{')[1:]:
        currency.append(i.split(',')[0].split(':')[1].replace('"', ''))
        timestamp_kst.append(datetime.utcfromtimestamp(
            int(soup.text.split(':{')[0].split(',')[2].split(':')[1].replace('"', ''))).strftime('%Y-%m-%d'))
        opening_price.append(i.split(',')[1].split(':')[1].replace('"', ''))
        high_price.append(i.split(',')[3].split(':')[1].replace('"', ''))
        low_price.append(i.split(',')[2].split(':')[1].replace('"', ''))
        close_price.append(i.split(',')[4].split(':')[1].replace('"', ''))
        acc_trade_value.append(float(i.split(',')[4].split(':')[1].replace('"', '')) * float(
            i.split(',')[5].split(':')[1].replace('"', '')))
        unit_traded.append(i.split(',')[5].split(':')[1].replace('"', ''))

    coinone_df = pd.concat([pd.DataFrame(currency, columns=['currency'])
                               , pd.DataFrame(timestamp_kst, columns=['timestamp_kst'])
                               , pd.DataFrame(opening_price, columns=['opening_price'])
                               , pd.DataFrame(high_price, columns=['high_price'])
                               , pd.DataFrame(low_price, columns=['low_price'])
                               , pd.DataFrame(close_price, columns=['close_price'])
                               , pd.DataFrame(acc_trade_value, columns=['acc_trade_value'])
                               , pd.DataFrame(unit_traded, columns=['unit_traded'])]
                           , axis=1)

    coinone_df['currency'] = [i.upper() for i in coinone_df['currency']]
    coinone_df['exchange'] = 'Coinone'
    print("Coinone Done!")

    return coinone_df



######################
### Korbit Crawler ###
######################
def korbit_crawler():
    """
    Korbit Crawl during 24H information
    """
    korbit_url = 'https://api.korbit.co.kr/v1/ticker/detailed/all'
    html = requests.get(korbit_url)
    soup = BeautifulSoup(html.text, "html.parser")

    currency = []
    timestamp_kst = []
    opening_price = []
    high_price = []
    low_price = []
    close_price = []
    acc_trade_value = []
    unit_traded = []

    for i in soup.text.split('},'):
        currency.append(i.split(':')[0].replace('{', '').replace('"', ''))
        timestamp_kst.append(datetime.utcfromtimestamp(int(i.split(':')[2].split(',')[0]) / 1000).strftime('%Y-%m-%d'))
        opening_price.append(i.split(',')[2].split(':')[1].replace('"', ''))
        high_price.append(i.split(',')[6].split(':')[1].replace('"', ''))
        low_price.append(i.split(',')[5].split(':')[1].replace('"', ''))
        close_price.append(i.split(',')[1].split(':')[1].replace('"', ''))
        acc_trade_value.append(float(i.split(',')[7].split(':')[1].replace('"', '')) * float(
            i.split(',')[1].split(':')[1].replace('"', '')))
        unit_traded.append(i.split(',')[7].split(':')[1].replace('"', ''))

    korbit_df = pd.concat([pd.DataFrame(currency, columns=['currency'])
                              , pd.DataFrame(timestamp_kst, columns=['timestamp_kst'])
                              , pd.DataFrame(opening_price, columns=['opening_price'])
                              , pd.DataFrame(high_price, columns=['high_price'])
                              , pd.DataFrame(low_price, columns=['low_price'])
                              , pd.DataFrame(close_price, columns=['close_price'])
                              , pd.DataFrame(acc_trade_value, columns=['acc_trade_value'])
                              , pd.DataFrame(unit_traded, columns=['unit_traded'])]
                          , axis=1)

    korbit_df['currency'] = [i.split('_')[0].upper() for i in korbit_df['currency']]
    korbit_df['exchange'] = 'Korbit'
    print("Korbit Done!")

    return korbit_df



#####################
### Gopax Crawler ###
#####################
def gopax_crawler():
    """
    Gopax Crawl during 24H information
    """
    gopax_url = 'https://api.gopax.co.kr/trading-pairs/stats'
    html = requests.get(gopax_url)
    soup = BeautifulSoup(html.text, "html.parser")

    currency = []
    timestamp_kst = []
    opening_price = []
    high_price = []
    low_price = []
    close_price = []
    acc_trade_value = []
    unit_traded = []

    for i in soup.text.split(',{'):
        currency.append(i.split(',')[0].split(':')[1].replace('"', ''))
        timestamp_kst.append(i.split(',')[6].split(':')[1].replace('"', '').replace('T', ' '))
        opening_price.append(i.split(',')[1].split(':')[1])
        high_price.append(i.split(',')[2].split(':')[1])
        low_price.append(i.split(',')[3].split(':')[1])
        close_price.append(i.split(',')[4].split(':')[1])
        acc_trade_value.append(float(i.split(',')[4].split(':')[1]) * float(i.split(',')[5].split(':')[1]))
        unit_traded.append(i.split(',')[5].split(':')[1])

    gopax_df = pd.concat([pd.DataFrame(currency, columns=['currency']),
                          pd.DataFrame(timestamp_kst, columns=['timestamp_kst']),
                          pd.DataFrame(opening_price, columns=['opening_price']),
                          pd.DataFrame(high_price, columns=['high_price']),
                          pd.DataFrame(low_price, columns=['low_price']),
                          pd.DataFrame(close_price, columns=['close_price']),
                          pd.DataFrame(acc_trade_value, columns=['acc_trade_value']),
                          pd.DataFrame(unit_traded, columns=['unit_traded'])],
                         axis=1)

    gopax_df['currency'] = [i.split('-')[0].upper() for i in gopax_df['currency']]
    gopax_df['exchange'] = 'Gopax'
    print("Gopax Done!")

    return gopax_df



#####################
### Upbit Crawler ###
#####################
def upbit_crawler():
    """
    Upbit Crawl during 24H information
    first, Crawling currency-pair
    second, Crawling 1H data
    third, Union 1H data(1H * 24)
    """
    # Crawling currency-pair
    upbit_url = 'https://api.upbit.com/v1/market/all'
    html = requests.get(upbit_url)
    soup = BeautifulSoup(html.text, "html.parser")

    upbit_currency = soup.text.split('},')

    ignore_char = '""'

    basis_dy = []
    exchange = []
    currency_pair = []
    currency_name = []

    for i in upbit_currency:
        currency_pair.append(i.split(':')[1].split(',')[0].replace('"', ''))
        currency_name.append(i.split(':')[3].replace('}]', '').replace('"', ''))

    currency_pair_df = pd.DataFrame(currency_pair)
    currency_name_df = pd.DataFrame(currency_name)

    upbit_pair = pd.concat([currency_pair_df, currency_name_df], axis=1)
    upbit_pair.columns = ['currency_pair', 'name']

    upbit_pair['currency_krw'] = [i.split('-')[0] for i in upbit_pair['currency_pair']]
    upbit_pair = upbit_pair.where(upbit_pair['currency_krw'] == 'KRW')
    upbit_pair = upbit_pair.dropna(axis=0)

    currency_m = []
    timestamp_kst_m = []
    opening_price_m = []
    high_price_m = []
    low_price_m = []
    close_price_m = []
    acc_trade_value_m = []
    unit_traded_m = []

    # crawling 1H data
    for i in upbit_pair['currency_pair']:
        upbit_price = 'https://api.upbit.com/v1/candles/minutes/60?market={}&count=24'.format(i)
        time.sleep(0.1)
        html = requests.get(upbit_price)
        soup = BeautifulSoup(html.text, 'html.parser')
        soup = soup.text.split(',{')

        for j in soup:
            currency_m.append(j.split(',')[0].split(':')[1].replace('"', ''))
            timestamp_kst_m.append(j.split(',')[1].split(':')[1].replace('T', ' ').replace('"', ''))
            opening_price_m.append(j.split(',')[3].split(':')[1])
            high_price_m.append(j.split(',')[4].split(':')[1])
            low_price_m.append(j.split(',')[5].split(':')[1])
            close_price_m.append(j.split(',')[6].split(':')[1])
            acc_trade_value_m.append(j.split(',')[8].split(':')[1])
            unit_traded_m.append(j.split(',')[9].split(':')[1])

    upbit_df_m = pd.concat([pd.DataFrame(currency_m, columns=['currency_m']),
                            pd.DataFrame(timestamp_kst_m, columns=['timestamp_kst_m']),
                            pd.DataFrame(opening_price_m, columns=['opening_price_m']),
                            pd.DataFrame(high_price_m, columns=['high_price_m']),
                            pd.DataFrame(low_price_m, columns=['low_price_m']),
                            pd.DataFrame(close_price_m, columns=['close_price_m']),
                            pd.DataFrame(acc_trade_value_m, columns=['acc_trade_value_m']),
                            pd.DataFrame(unit_traded_m, columns=['unit_traded_m'])],
                           axis=1)

    upbit_df_m['acc_trade_value_m'] = upbit_df_m['acc_trade_value_m'].astype(float)
    upbit_df_m['unit_traded_m'] = upbit_df_m['unit_traded_m'].astype(float)

    # Union 1H data
    currency = []
    timestamp_kst = []
    opening_price = []
    high_price = []
    low_price = []
    close_price = []
    acc_trade_value = []
    unit_traded = []

    currency.append(upbit_df_m.groupby('currency_m').max().reset_index(drop=False)['currency_m'])
    timestamp_kst.append(upbit_df_m.groupby('currency_m').max()['timestamp_kst_m'])
    opening_price.append(upbit_df_m.groupby('currency_m').min()['opening_price_m'])
    high_price.append(upbit_df_m.groupby('currency_m').max()['high_price_m'])
    low_price.append(upbit_df_m.groupby('currency_m').min()['low_price_m'])
    close_price.append(upbit_df_m.groupby('currency_m').max()['close_price_m'])
    acc_trade_value.append(upbit_df_m.groupby('currency_m').sum('acc_trade_value_m')['acc_trade_value_m'])
    unit_traded.append(upbit_df_m.groupby('currency_m').sum('unit_traded_m')['unit_traded_m'])

    upbit_df = pd.concat([pd.DataFrame(currency).T.set_index('currency_m'),
                          pd.DataFrame(timestamp_kst).T,
                          pd.DataFrame(opening_price).T,
                          pd.DataFrame(high_price).T,
                          pd.DataFrame(low_price).T,
                          pd.DataFrame(close_price).T,
                          pd.DataFrame(acc_trade_value).T,
                          pd.DataFrame(unit_traded).T],
                         axis=1)

    upbit_df = upbit_df.reset_index(drop=False)
    upbit_df.columns = ['currency', 'timestamp_kst', 'opening_price', 'high_price', 'low_price', 'close_price',
                        'acc_trade_value', 'unit_traded']

    upbit_df['currency'] = [i.split('-')[1] for i in upbit_df['currency']]
    upbit_df['exchange'] = 'Upbit'
    print("Upbit Done!")

    return upbit_df



#####################
### Summary Table ###
#####################
def MakeDataFrame(upbit_df, bithumb_df, coinone_df, korbit_df, gopax_df):
    """
    Make Total DataFrame
    """
    total_df = pd.concat([bithumb_df, coinone_df, korbit_df, upbit_df, gopax_df], axis=0)
    total_df.columns = ['base_currency', 'etime_date', 'open_price', 'high_price', 'low_price', 'close_price',
                        'krw_volume', 'base_volume', 'exchange']
    total_df = total_df[
        ['exchange', 'base_currency', 'open_price', 'high_price', 'low_price', 'close_price', 'krw_volume',
         'base_volume', 'etime_date']]
    total_df = total_df.reset_index(drop=True)
    print(total_df)
    print("All Done!")
    logging.info("-------------sys.argv list-------------", print(sys.argv), sys.argv)

    # return total_df # Original Method
    return total_df.to_csv('/Users/yoo/Data-dev/nlp/reference/DataFrame/price.csv') # Temporary save DataFrame



##############################
### Upload DataFrame to S3 ###
##############################
# def DataFrameToS3Upload(total_df):
#     """
#     Upload DataFrame(df) to S3 bucket
#     """
#     try:
#         result = total_df.to_dict()
#         results_string = str(total_df.T.to_dict().values()).replace("dict_values([", '').replace('])', '').replace("'",
#                                                                                                                    '"').replace(
#             ' ', '').replace('},{', '}{')
#         logging.info(results_string[:51])
#         logging.info('******************* Changing JSON FORMAT to Bytes *******************')
#         results_bytes = results_string.encode('utf-8')
#         logging.info('******************* gzip the results *******************')
#         results_gzip = gzip.compress(results_bytes)
#         results = results_gzip
#
#         dag_id = 'airflow_public_api_exchange_market_pairs_to_s3_v0002'
#
#         s3_key = 'ods_external/airflow_public_api_exchange_market_pairs_to_s3_v0002/' + str(
#             (datetime.today() + timedelta(days=-1)).strftime('%Y%m%d')).replace('-', '') + '.json.gzip'
#         s3_hook = AwsHook(aws_conn_id='{{ params.s3_conn_id }}')
#         s3_client_type = s3_hook.get_client_type(client_type='s3', region_name='')
#         s3 = s3_client_type
#         s3.put_object(
#             Bucket=s3_bucket,
#             Body=results,
#             Key=s3_key
#         )
#         print("--------------- Python process end ---------------")
#
#     except Exception as e:
#         print("S3 uploader Error", e)
#
#     return results



####################
### Main Complie ###
####################
df = pd.DataFrame(MakeDataFrame(
    upbit_df=upbit_crawler(),
    bithumb_df=bithumb_crawler(),
    coinone_df=coinone_crawler(),
    korbit_df=korbit_crawler(),
    gopax_df=gopax_crawler()))

# DataFrameToS3Upload(df)