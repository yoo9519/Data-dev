# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import requests
import re
import sys
print(sys.executable)
from bs4 import BeautifulSoup
import logging
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
import os
from io import StringIO
import boto3
import time
import shutil


### Use list ###
basic_date_list = []
basic_news_article = []
depth_date_list = []
depth_news_article = []
clear_word = []


### Crawler 1 ###
def basic_crawler():
    """
    Crawler(data gathering)
    """
    d = []
    a = []

    for i in tqdm(range(1,21), desc="Extracting Web page.."):
        url = 'https://www.coindeskkorea.com/news/articleList.html?page={}&total=6048&box_idxno=&view_type=sm'.format(i)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'lxml')

        for j in range(1,21):
            date = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(j))
            article = soup.select(
            '#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(
                j))

            d.append(date)
            a.append(article)

    for _ in d:
        if _:
            basic_date_list.append(_)

    for _ in a:
        if _:
            basic_news_article.append(_)

    return basic_date_list, basic_news_article


### Crawler 2 ###
def depth_crawler():
    """
    Optimization basic_crawl data
    """
    # depth_news_article
    processing_news = []
    for i in tqdm(basic_news_article, desc="Extracting News url.."):
        url = 'https://www.coindeskkorea.com'+str(i).split(' ')[2].split('"')[1]

        processing_news.append(url)

    for i in tqdm(processing_news, desc="Crawling News article.."):
        resq = requests.get(i)
        soup = BeautifulSoup(resq.text,'lxml')
        n = str(soup.find_all('p')).split('<p><a href=')[0]

        depth_news_article.append(n)

    # depth_date_list
    for i in tqdm(basic_date_list, desc="Crawling News datetime.."):
        d = i[0].text.split(' ')[-2]

        depth_date_list.append(d)

    return depth_news_article, depth_date_list


### pre-processing ###
def processing():
    """
    Article pre-processing
    """
    process = []
    for i in tqdm(depth_news_article, desc="processing complie 1"):
        a = re.compile('[가-힣]+').findall(i)

        process.append(a)

    for i in tqdm(process, desc="processing complie 2"):
        a = ' '.join(i)
        b = re.compile('[제보-보내주세요]').sub(' ',a)
        # c = b.replace('도자료','')

        clear_word.append(' '+b)

    return clear_word


### dataframe ###
def dataframe():
    df_article = pd.DataFrame(clear_word, columns=['article'])
    df_date = pd.DataFrame(depth_date_list, columns=['created_date'])
    df_total = pd.concat([df_date, df_article], axis=1)
    df_total = df_total.groupby('created_date').sum(str(df_total['article']))
    print("Make DataFrame Structure")

    return df_total.to_csv('/Users/yoo/Data-dev/nlp/reference/DataFrame/article.csv')


basic_crawler()
depth_crawler()
processing()
dataframe()
print("Done")