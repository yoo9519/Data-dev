# -*- coding: utf-8 -*-

"""
team: Data
Author: Cline Yoo
* created: 2021-07-14

This is the file for operating Bash Script.
"""

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime, timedelta
from airflow.contrib.hooks.aws_hook import AwsHook
from airflow.models import Variable
from airflow.hooks.S3_hook import S3Hook
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

airflow_home = os.environ['AIRFLOW_HOME']


logging.info(print(sys.argv))

pd.options.display.float_format = '{:.5f}'.format

# Upbit API
upbit_url = 'https://api.upbit.com/v1/market/all'
html = requests.get(upbit_url)
soup = BeautifulSoup(html.text, "html.parser")


upbit_currency = soup.text.split('},')

print(upbit_currency[0].split(':')[1].split(',')[0], upbit_currency[0].split(':')[3])


ignore_char = '""'

basis_dy = []
exchange = []
currency_pair = []
currency_name = []

for i in upbit_currency:
    currency_pair.append(i.split(':')[1].split(',')[0].replace('"', ''))
    currency_name.append(i.split(':')[3].replace('}]', '').replace('"', ''))

# check upbit {currency}_{counter} pair

currency_pair_df = pd.DataFrame(currency_pair)
currency_name_df = pd.DataFrame(currency_name)

upbit_pair = pd.concat([currency_pair_df,currency_name_df],axis=1)
upbit_pair.columns = ['currency_pair','name']

# {currency}_krw pair check
upbit_pair['currency_krw'] = [i.split('-')[0] for i in upbit_pair['currency_pair']]
upbit_pair = upbit_pair.where(upbit_pair['currency_krw'] == 'KRW')
upbit_pair = upbit_pair.dropna(axis=0)
print(upbit_pair)


# Upbit candle chart crawling
currency = []
timestamp_kst = []
opening_price = []
high_price = []
low_price = []
acc_trade_value = []
unit_traded = []

for i in upbit_pair['currency_pair']:
    upbit_price = 'https://api.upbit.com/v1/candles/days/?market={}&count=1'.format(i)
    time.sleep(0.5)
    html = requests.get(upbit_price)
    soup = BeautifulSoup(html.text, "html.parser")
    currency.append(soup.text.split(',{')[0].split(':')[1].split(',')[0].replace('"', ''))
    timestamp_kst.append(soup.text.split(',{')[0].split(',')[1].split(':')[1].replace('"','').replace('T00',' '))
    opening_price.append(soup.text.split(',{')[0].split(',')[3].split(':')[1].replace('"', ''))
    high_price.append(soup.text.split(',{')[0].split(',')[4].split(':')[1].replace('"', ''))
    low_price.append(soup.text.split(',{')[0].split(',')[5].split(':')[1].replace('"', ''))
    #     trade_price.append(soup.text.split(',{')[0].split(',')[6].split(':')[1].replace('"',''))
    acc_trade_value.append(soup.text.split(',{')[0].split(',')[8].split(':')[1].replace('"', ''))
    unit_traded.append(soup.text.split(',{')[0].split(',')[9].split(':')[1].replace('"', ''))

print(upbit_url,html)
print(currency[0])
print(timestamp_kst[0])
print(opening_price[0])
print(high_price[0])
print(low_price[0])
# print(trade_price[0])
print(acc_trade_value[0])
print(unit_traded[0])


upbit_df = pd.concat([pd.DataFrame(currency,columns=['currency'])
                         ,pd.DataFrame(timestamp_kst,columns=['timestamp_kst'])
                         ,pd.DataFrame(opening_price,columns=['opening_price'])
                         ,pd.DataFrame(high_price,columns=['high_price'])
                         ,pd.DataFrame(low_price,columns=['low_price'])
                         ,pd.DataFrame(acc_trade_value,columns=['acc_trade_value'])
                         ,pd.DataFrame(unit_traded,columns=['unit_traded'])]
                     ,axis=1)

upbit_df['currency'] = [i.split('-')[1] for i in upbit_df['currency']]
upbit_df['exchange'] = 'Upbit'

print(upbit_df)


# Bithumb API
bithumb_url = 'https://api.bithumb.com/public/ticker/ALL_KRW'
html = requests.get(bithumb_url)
soup = BeautifulSoup(html.text,"html.parser")

print(bithumb_url, html)

soup = soup.text.replace('{"status":"0000","data":{', '')

currency = []
timestamp_kst = []
opening_price = []
high_price = []
low_price = []
acc_trade_value = []
unit_traded = []


for i in soup.split('},')[:-1]:
    currency.append(i.split(':')[0].replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(soup.split('},')[-1].replace('"','').replace('}}','').replace('date:',''))/1000).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[0].split(':')[2].replace('"',''))
    high_price.append(i.split(',')[3].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    acc_trade_value.append(i.split(',')[5].split(':')[1].replace('"',''))
    unit_traded.append(i.split(',')[4].split(':')[1].replace('"',''))


bithumb_df = pd.concat([pd.DataFrame(currency,columns=['currency'])
                           ,pd.DataFrame(timestamp_kst,columns=['timestamp_kst'])
                           ,pd.DataFrame(opening_price,columns=['opening_price'])
                           ,pd.DataFrame(high_price,columns=['high_price'])
                           ,pd.DataFrame(low_price,columns=['low_price'])
                           ,pd.DataFrame(acc_trade_value,columns=['acc_trade_value'])
                           ,pd.DataFrame(unit_traded,columns=['unit_traded'])]
                       ,axis=1)

bithumb_df['exchange'] = 'Bithumb'

print(bithumb_df)
print(currency[0])
print(timestamp_kst[0])
print(opening_price[0])
print(high_price[0])
print(low_price[0])
print(acc_trade_value[0])
print(unit_traded[0])


# Coinone API
coinone_url = 'https://api.coinone.co.kr/ticker?currency=all'
html = requests.get(coinone_url)
soup = BeautifulSoup(html.text,"html.parser")

print(coinone_url, html)

currency = []
timestamp_kst = []
opening_price = []
high_price = []
low_price = []
acc_trade_value = []
unit_traded = []

for i in soup.text.split(':{')[1:]:
    currency.append(i.split(',')[0].split(':')[1].replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(soup.text.split(':{')[0].split(',')[2].split(':')[1].replace('"',''))).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[1].split(':')[1].replace('"',''))
    high_price.append(i.split(',')[3].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    acc_trade_value.append(float(i.split(',')[4].split(':')[1].replace('"',''))*float(i.split(',')[5].split(':')[1].replace('"','')))
    unit_traded.append(i.split(',')[5].split(':')[1].replace('"',''))

print(currency[1])
print(timestamp_kst[1])
print(opening_price[1])
print(high_price[1])
print(low_price[1])
print(acc_trade_value[1])
print(unit_traded[1])

coinone_df = pd.concat([pd.DataFrame(currency, columns=['currency'])
                           ,pd.DataFrame(timestamp_kst, columns=['timestamp_kst'])
                           ,pd.DataFrame(opening_price, columns=['opening_price'])
                           ,pd.DataFrame(high_price, columns=['high_price'])
                           ,pd.DataFrame(low_price, columns=['low_price'])
                           ,pd.DataFrame(acc_trade_value, columns=['acc_trade_value'])
                           ,pd.DataFrame(unit_traded, columns=['unit_traded'])]
                       ,axis=1)

coinone_df['currency'] = [i.upper() for i in coinone_df['currency']]
coinone_df['exchange'] = 'Coinone'
print(coinone_df)


# Korbit API
korbit_url = 'https://api.korbit.co.kr/v1/ticker/detailed/all'
html = requests.get(korbit_url)
soup = BeautifulSoup(html.text,"html.parser")

currency = []
timestamp_kst = []
opening_price = []
high_price = []
low_price = []
acc_trade_value = []
unit_traded = []


for i in soup.text.split('},'):
    currency.append(i.split(':')[0].replace('{','').replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(i.split(':')[2].split(',')[0])/1000).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    high_price.append(i.split(',')[6].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[5].split(':')[1].replace('"',''))
    acc_trade_value.append(float(i.split(',')[7].split(':')[1].replace('"',''))*float(i.split(',')[1].split(':')[1].replace('"','')))
    unit_traded.append(i.split(',')[7].split(':')[1].replace('"',''))

print(korbit_url, html)
print(currency[1])
print(timestamp_kst[1])
print(opening_price[1])
print(high_price[1])
print(low_price[1])
print(acc_trade_value[1])
print(unit_traded[1])


korbit_df = pd.concat([pd.DataFrame(currency,columns=['currency'])
                          ,pd.DataFrame(timestamp_kst,columns=['timestamp_kst'])
                          ,pd.DataFrame(opening_price,columns=['opening_price'])
                          ,pd.DataFrame(high_price,columns=['high_price'])
                          ,pd.DataFrame(low_price,columns=['low_price'])
                          ,pd.DataFrame(acc_trade_value,columns=['acc_trade_value'])
                          ,pd.DataFrame(unit_traded,columns=['unit_traded'])]
                      ,axis=1)


korbit_df['currency'] = [i.split('_')[0].upper() for i in korbit_df['currency']]
korbit_df['exchange'] = 'Korbit'
print(korbit_df)


# total_df
total_df = pd.concat([upbit_df,bithumb_df,coinone_df,korbit_df],axis=0)
total_df.columns = ['base_currency','etime_at','open_price','high_price','low_price','krw_volume','base_volume','exchange']
total_df = total_df.reset_index(drop=True)
print(total_df)
logging.info("-------------sys.argv list-------------", print(sys.argv), sys.argv)


############################################################################################################################
################################################## Upload procedure to S3 ##################################################
############################################################################################################################


# total_df.to_csv(index=False) to s3
s3_sensorflag_bucket = 'Security'
s3_conn_id = 'Security'
s3_bucket = 'Security'
jsonpaths_bucket = 'Security'

# s3_key = 'Schema/' + dag_id + "/{{ (execution_date + macros.timedelta(days=1)).strftime('%Y%m%d') }}"
#
# dag_id = 'Your DAG Name'
# s3_key = 'Schema/' + dag_id + "/'{{ tomorrow_ds_nodash }}'" + '.csv'
# csv_buffer = StringIO()
# total_df.to_csv(csv_buffer)
# s3_resource = boto3.resource('s3')
# s3_resource.Object(s3_bucket, 'total_df.csv').put(Body=csv_buffer.getvalue())
# logging.info("------------------------ Finish: uploading csv file process ------------------------")


# total_df.json.gzip to s3
result = total_df.to_dict()
results_string = str(total_df.T.to_dict().values()).replace("dict_values([",'').replace('])','').replace("'",'"').replace(' ','').replace('},{','}{')
logging.info(results_string[:51])
logging.info('******************* Changing JSON FORMAT to Bytes *******************')
results_bytes = results_string.encode('utf-8')
logging.info('******************* gzip the results *******************')
results_gzip = gzip.compress(results_bytes)
results = results_gzip

logging.info(total_df.head())


dag_id = 'Your DAG Name'
s3_key = 'Schema/Your DAG Name and version/' + str((datetime.today() + timedelta(days=-1)).strftime('%Y%m%d')).replace('-','') + '.json.gzip'

s3_hook = AwsHook(aws_conn_id='{{ params.s3_conn_id }}')
s3_client_type = s3_hook.get_client_type(client_type='s3', region_name='Security')
s3 = s3_client_type
s3.put_object(
    Bucket=s3_bucket,
    Body=results,
    Key=s3_key
)

logging.info("--------------- Python process end ---------------")
# S3 -> Data Lake