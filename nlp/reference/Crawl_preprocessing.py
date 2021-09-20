import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
from tqdm.notebook import tqdm
import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)
import transformers as ts
print(ts.__version__)
from datetime import datetime
import logging

import psycopg2 as pg
import getpass

### main method - coindesk(cryptocurrency newsdesk) ###
### article link to article ###
### Make Crawler and pre-processing ###

date_list = []
news_article = []

for i in tqdm(range(1, 2)):
    url = 'https://www.coindeskkorea.com/news/articleList.html?page={}&total=6048&box_idxno=&view_type=sm'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')

    for j in range(1, 21):
        date = soup.select(
            '#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(
                j))
        article = soup.select(
            '#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(
                j))

        date_list.append(date)
        news_article.append(article)

print("total: {} date".format(len(date_list)))
print("total: {} article".format(len(news_article)))

processing_news = []

for i in news_article:
    na = 'https://www.coindeskkorea.com' + str(i).split(' ')[2].split('"')[1]

    processing_news.append(na)


real_news = []

for i in tqdm(processing_news):
    resq = requests.get(i)
    soup = BeautifulSoup(resq.text, 'lxml')
    #     n = str(soup.find_all('p')).replace('<p>','').replace('<br/>\r\n','').replace('●','').replace('</p>','').replace('<strong>■','').replace('\xa0','').replace('<br/>\n','')
    n = str(soup.find_all('p')).split('<p><a href=')[0]
    real_news.append(n)

print(real_news[:5])

created_at = []

for i in date_list:
    d = i[0].text.split(' ')[-2]
    created_at.append(d)

print(created_at[:5])
logging.info('***************************************************************************************')
logging.info('******************************** News article is ready ********************^***********')
logging.info('************************* Now, Article will be pre-processing *************************')


df_news = []

for i in real_news:
    a = re.compile('[가-힣]+').findall(i)
    df_news.append(a)

created_at = []

for i in date_list:
    d = i[0].text.split(' ')[-2]
    created_at.append(d)

clear_word = []

for i in df_news:
    c = ' '.join(i)
    d = re.compile('[제보-보내주세요]').sub(' ',c)
    d = d.replace('도자료','')
    clear_word.append(' '+d)


df_word = pd.DataFrame(clear_word,columns=['word'])
df_date = pd.DataFrame(created_at,columns=['created_date'])
total_df = pd.concat([df_date,df_word],axis=1)
total_df = total_df.groupby('created_date').sum(str(total_df['word']))


total_df.to_csv('/Users/yoo/Data-dev/nlp/article/total_df.csv')


### Load via DataBase ###
### CMC -> DB -> Load ###
# db_con = psycopg2.connect(dbname=dbname,
#                           host=host,
#                           port=post,
#                           user=user,
#                           password=password
#                           )


# query = f"""
# select trade_date, last_price
# from currency_price
# where trade_date >= '2018-04-13'
# and split_part(listed,'_',1) = input('')
# and split_part(listed,'_',2) = 'krw'
# order by trade_date
# """


### Labeling ###
# price_df = pd.read_clipboard()
# price_df = pd.read_sql(query, db_con)
# price_df = price_df[['trade_date','latest_trade_price']]
# price_df = price_df.set_index('trade_date')
#
# df = pd.concat([price_df,total_df],axis=1)
# df['y_price'] = df['latest_trade_price'].shift(-1)
# df['label'] = df['latest_trade_price'] < df['y_price']
# df['label'] = df['label'].replace(False,0).replace(True,1)
#
# f_df = df[['word','label']]
# f_df.to_csv('f_df.csv')           # Saved to CSV File


# 2021.09.18 -- git pust test