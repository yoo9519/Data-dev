import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
from tqdm.notebook import tqdm
import sys
print(sys.executable)
from datetime import datetime
import logging


### crawler
def crawler(url):
    date_list = []
    news_article = []

    for i in tqdm(range(1,2)):
        url = 'https://www.coindeskkorea.com/news/articleList.html?page={}&total=6048&box_idxno=&view_type=sm'.format(i)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'lxml')

        for j in range(1,21):
            date = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(j))
            article = soup.select(
            '#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(
                j))

            date_list.append(date)
            news_article.append(article)

    return date_list, news_article


### news_preprocessing
def processing(url):
    processing_news = []

    pass



def crypto_price():
    pass