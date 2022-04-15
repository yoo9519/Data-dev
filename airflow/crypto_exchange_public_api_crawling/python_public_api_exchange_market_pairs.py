# -*- coding: utf-8 -*-
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
close_price = []
acc_trade_value = []
unit_traded = []


for i in soup.split('},')[:-1]:
    currency.append(i.split(':')[0].replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(soup.split('},')[-1].replace('"','').replace('}}','').replace('date:',''))/1000).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[0].split(':')[2].replace('"',''))
    high_price.append(i.split(',')[3].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    close_price.append(i.split(',')[1].split(':')[1].replace('"',''))
    acc_trade_value.append(i.split(',')[5].split(':')[1].replace('"',''))
    unit_traded.append(i.split(',')[4].split(':')[1].replace('"',''))


bithumb_df = pd.concat([pd.DataFrame(currency,columns=['currency'])
                           ,pd.DataFrame(timestamp_kst,columns=['timestamp_kst'])
                           ,pd.DataFrame(opening_price,columns=['opening_price'])
                           ,pd.DataFrame(high_price,columns=['high_price'])
                           ,pd.DataFrame(low_price,columns=['low_price'])
                           ,pd.DataFrame(close_price,columns=['close_price'])
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
print(close_price[0])
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
close_price = []
acc_trade_value = []
unit_traded = []

for i in soup.text.split(':{')[1:]:
    currency.append(i.split(',')[0].split(':')[1].replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(soup.text.split(':{')[0].split(',')[2].split(':')[1].replace('"',''))).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[1].split(':')[1].replace('"',''))
    high_price.append(i.split(',')[3].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    close_price.append(i.split(',')[4].split(':')[1].replace('"',''))
    acc_trade_value.append(float(i.split(',')[4].split(':')[1].replace('"',''))*float(i.split(',')[5].split(':')[1].replace('"','')))
    unit_traded.append(i.split(',')[5].split(':')[1].replace('"',''))

print(currency[1])
print(timestamp_kst[1])
print(opening_price[1])
print(high_price[1])
print(low_price[1])
print(close_price[1])
print(acc_trade_value[1])
print(unit_traded[1])

coinone_df = pd.concat([pd.DataFrame(currency, columns=['currency'])
                           ,pd.DataFrame(timestamp_kst, columns=['timestamp_kst'])
                           ,pd.DataFrame(opening_price, columns=['opening_price'])
                           ,pd.DataFrame(high_price, columns=['high_price'])
                           ,pd.DataFrame(low_price, columns=['low_price'])
                           ,pd.DataFrame(close_price, columns=['close_price'])
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
close_price = []
acc_trade_value = []
unit_traded = []


for i in soup.text.split('},'):
    currency.append(i.split(':')[0].replace('{','').replace('"',''))
    timestamp_kst.append(datetime.utcfromtimestamp(int(i.split(':')[2].split(',')[0])/1000).strftime('%Y-%m-%d'))
    opening_price.append(i.split(',')[2].split(':')[1].replace('"',''))
    high_price.append(i.split(',')[6].split(':')[1].replace('"',''))
    low_price.append(i.split(',')[5].split(':')[1].replace('"',''))
    close_price.append(i.split(',')[1].split(':')[1].replace('"',''))
    acc_trade_value.append(float(i.split(',')[7].split(':')[1].replace('"',''))*float(i.split(',')[1].split(':')[1].replace('"','')))
    unit_traded.append(i.split(',')[7].split(':')[1].replace('"',''))

print(korbit_url, html)
print(currency[1])
print(timestamp_kst[1])
print(opening_price[1])
print(high_price[1])
print(low_price[1])
print(close_price[1])
print(acc_trade_value[1])
print(unit_traded[1])


korbit_df = pd.concat([pd.DataFrame(currency,columns=['currency'])
                          ,pd.DataFrame(timestamp_kst,columns=['timestamp_kst'])
                          ,pd.DataFrame(opening_price,columns=['opening_price'])
                          ,pd.DataFrame(high_price,columns=['high_price'])
                          ,pd.DataFrame(low_price,columns=['low_price'])
                          ,pd.DataFrame(close_price,columns=['close_price'])
                          ,pd.DataFrame(acc_trade_value,columns=['acc_trade_value'])
                          ,pd.DataFrame(unit_traded,columns=['unit_traded'])]
                      ,axis=1)


korbit_df['currency'] = [i.split('_')[0].upper() for i in korbit_df['currency']]
korbit_df['exchange'] = 'Korbit'
print(korbit_df)




# Gopax API
gopax_url = 'https://api.gopax.co.kr/trading-pairs/stats'
html = requests.get(gopax_url)
soup = BeautifulSoup(html.text,"html.parser")


currency = []
timestamp_kst = []
opening_price = []
high_price = []
low_price = []
close_price = []
acc_trade_value = []
unit_traded = []

for i in soup.text.split(',{'):
    currency.append(i.split(',')[0].split(':')[1].replace('"',''))
    timestamp_kst.append(i.split(',')[6].split(':')[1].replace('"','').replace('T',' '))
    opening_price.append(i.split(',')[1].split(':')[1])
    high_price.append(i.split(',')[2].split(':')[1])
    low_price.append(i.split(',')[3].split(':')[1])
    close_price.append(i.split(',')[4].split(':')[1])
    acc_trade_value.append(float(i.split(',')[4].split(':')[1])*float(i.split(',')[5].split(':')[1]))
    unit_traded.append(i.split(',')[5].split(':')[1])


gopax_df = pd.concat([pd.DataFrame(currency,columns=['currency']),
                      pd.DataFrame(timestamp_kst,columns=['timestamp_kst']),
                      pd.DataFrame(opening_price,columns=['opening_price']),
                      pd.DataFrame(high_price,columns=['high_price']),
                      pd.DataFrame(low_price,columns=['low_price']),
                      pd.DataFrame(close_price,columns=['close_price']),
                      pd.DataFrame(acc_trade_value,columns=['acc_trade_value']),
                      pd.DataFrame(unit_traded,columns=['unit_traded'])],
                     axis=1)


gopax_df['currency'] = [i.split('-')[0].upper() for i in gopax_df['currency']]
gopax_df['exchange'] = 'Gopax'
# gopax_df.columns = ['base_currency','date_checker','open_price','high_price','low_price','close_price','krw_volume','base_volume','exchange']
print(gopax_df)



# Upbit
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

upbit_pair = pd.concat([currency_pair_df,currency_name_df],axis=1)
upbit_pair.columns = ['currency_pair','name']


upbit_pair['currency_krw'] = [i.split('-')[0] for i in upbit_pair['currency_pair']]
upbit_pair = upbit_pair.where(upbit_pair['currency_krw']=='KRW')
upbit_pair = upbit_pair.dropna(axis=0)


currency_m = []
timestamp_kst_m = []
opening_price_m = []
high_price_m = []
low_price_m = []
close_price_m = []
acc_trade_value_m = []
unit_traded_m = []

for i in upbit_pair['currency_pair']:
    upbit_price = 'https://api.upbit.com/v1/candles/minutes/60?market={}&count=24'.format(i)
    time.sleep(0.15)
    html = requests.get(upbit_price)
    soup = BeautifulSoup(html.text, 'html.parser')

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
print(upbit_df_m.shape)


upbit_df_m['acc_trade_value_m'] = upbit_df_m['acc_trade_value_m'].astype(float)
upbit_df_m['unit_traded_m'] = upbit_df_m['unit_traded_m'].astype(float)



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
upbit_df.columns=['currency','timestamp_kst','opening_price','high_price','low_price','close_price','acc_trade_value','unit_traded']


upbit_df['currency'] = [i.split('-')[1] for i in upbit_df['currency']]
upbit_df['exchange'] = 'Upbit'

print(upbit_df)



# total_df
total_df = pd.concat([upbit_df,bithumb_df,coinone_df,korbit_df,gopax_df],axis=0)
total_df.columns = ['base_currency','etime_date','open_price','high_price','low_price','close_price','krw_volume','base_volume','exchange']
total_df = total_df[['exchange','base_currency','open_price','high_price','low_price','close_price','krw_volume','base_volume','etime_date']]
total_df = total_df.reset_index(drop=True)
print(total_df)
logging.info("-------------sys.argv list-------------", print(sys.argv), sys.argv)


############################################################################################################################
################################################## Upload procedure to S3 ##################################################
############################################################################################################################


# total_df.to_csv(index=False) to s3
s3_sensorflag_bucket = Security
s3_conn_id = Security
s3_bucket = Security
jsonpaths_bucket = Security

# s3_key = 'Data Warehouse/' + dag_id + "/{{ (execution_date + macros.timedelta(days=1)).strftime('%Y%m%d') }}"
#
# dag_id = 'Your Dag Name and version'
# s3_key = 'Data Warehouse/' + dag_id + "/'{{ tomorrow_ds_nodash }}'" + '.csv'
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


dag_id = 'Your Dag Name and version'
s3_key = 'Data Warehouse/Your Dag Name and version/' + str((datetime.today() + timedelta(days=-1)).strftime('%Y%m%d')).replace('-','') + '.json.gzip'

s3_hook = AwsHook(aws_conn_id='{{ params.s3_conn_id }}')
s3_client_type = s3_hook.get_client_type(client_type='s3', region_name=Security)
s3 = s3_client_type
s3.put_object(
    Bucket=s3_bucket,
    Body=results,
    Key=s3_key
)

logging.info("--------------- Python process end ---------------")
