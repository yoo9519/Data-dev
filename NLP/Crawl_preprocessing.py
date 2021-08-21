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


# main method - coindesk(cryptocurrency newsdesk)
# article link to article

date_list = []
news_article = []

for i in tqdm(range(1, 387)):
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
