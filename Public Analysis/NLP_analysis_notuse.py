#!/usr/bin/env python
# coding: utf-8

# # Predict to BITCOIN price/volume by Sentimental Analysis(NLP)
# ## create by Cline
# ----

# ### 필요모듈 import
# - ekonlpy, konlpy, mecab ..

# In[71]:


get_ipython().run_line_magic('cd', '/tmp')


# In[72]:


get_ipython().system('ls')


# In[73]:


get_ipython().system('curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz')


# In[76]:


get_ipython().system('curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz')


# In[8]:


get_ipython().system('pip install ./eKoNLPy')


# In[ ]:


get_ipython().system('sudo apt-get install g++ openjdk-7-jdk')
get_ipython().system('sudo apt-get install python-dev; pip install konlpy')
get_ipython().system('sudo apt-get install python3-dev; pip3 install konlpy')
get_ipython().system('sudo apt-get install curl')
get_ipython().system('bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)')


# In[3]:


from io import StringIO
from io import open

import pandas as pd
import seaborn as sns
from krwordrank.word import KRWordRank
import numpy as np
from collections import defaultdict
import re
import requests

from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
from ekonlpy.sentiment import MPCK
mpck = MPCK()
from datetime import datetime, timedelta
from tqdm import tqdm


# In[5]:


# !bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)


# ---
# ### 데이터 1 : 코인데스크코리아

# In[7]:


# 2 method

def get_text(url):
    source_from_url = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_from_url, 'lxml', from_encoding='utf-8')
    text = ''
    
    for item in soup.find_all('div', id='list_news'):
        text = text + str(item.find_all(text=True))
        
        return text


# user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type(2) div.text-block p a
# 
# user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type(1) div.text-block p a
# 

# In[2]:


# main method

news_title = []
date_list = []
news_article = []

for i in tqdm(range(1, 303)):
    url = 'https://www.coindeskkorea.com/news/articleList.html?page={}&total=6048&box_idxno=&view_type=sm'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    
    for j in range(1, 121):
        title = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-titles a strong'.format(j))
        date = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(j))
        article = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(j))
        
        news_title.append(title)
        date_list.append(date)
        news_article.append(article)
        
print("총 {}개의 전처리가 필요한 제목".format(len(news_title)))
print("총 {}개의 전처리가 필요한 날짜".format(len(date_list)))
print("총 {}개의 전처리가 필요한 기사".format(len(news_article)))


# In[3]:


news_title[:7]


# In[4]:


clear_title = []

for title in tqdm(news_title):
    for tit in title:
        for t in tit:
#             print(t)
            clear_title.append(t)
            
print("전처리 후 제목: {}".format(len(clear_title)))
clear_title[:10]


# In[5]:


normal_date = []

for date_ in tqdm(date_list):
    for date in date_:
        for d in date:
            normal_date.append(d)
            
normal_date = normal_date[1::2]
print("전처리 후 제목: {}".format(len(normal_date)))
normal_date[:10]


# In[6]:


clear_date = []

for _date in normal_date:
    clear_date.append(_date[1:-6])
    
clear_date[:5]


# In[7]:


source = []

for news in tqdm(news_article):
    for new in news:
        for nw in new:
            source.append(nw)
            
print(len(source))
print(source[:5])


clear_article = []

for i in tqdm(source):
    clear_article.append(i[6:-5])

print("전처리 후 제목: {}".format(len(clear_article)))
clear_article[:5]


# In[10]:


date = pd.DataFrame(clear_date)
title = pd.DataFrame(clear_title)
article = pd.DataFrame(clear_article)

corpus = pd.concat([date, title, article], axis=1)
corpus.columns = ['timestamp', 'title', 'article']
corpus


# In[18]:


# corpus.to_csv('corpus.csv')


# ### 데이터 2 : 네이버 뉴스

# 'https://search.naver.com/search.naver?&where=news&query=%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2018.03.01&de=2021.01.22&docid=&nso=so:r,p:from20180301to20210122,a:all&mynews=0&cluster_rank=46&start={}&refresh_start=0'.format(i)

# In[15]:


# main method

naver_title = []
naver_date = []
naver_article = []

for i in tqdm(range(1, 303)):
    url = 'https://search.naver.com/search.naver?&where=news&query=%EB%B9%84%ED%8A%B8%EC%BD%94%EC%9D%B8&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2018.03.01&de=2021.01.22&docid=&nso=so:r,p:from20180301to20210122,a:all&mynews=0&cluster_rank=46&start={}&refresh_start=0'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    
    for j in range(1, 21):
        title = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-titles a strong'.format(j))
        date = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(j))
        article = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(j))
        
        news_title.append(title)
        date_list.append(date)
        news_article.append(article)
        
print("총 {}개의 전처리가 필요한 제목".format(len(news_title)))
print("총 {}개의 전처리가 필요한 날짜".format(len(date_list)))
print("총 {}개의 전처리가 필요한 기사".format(len(news_article)))


# ---

# ### 비트코인 가격 수집

# In[20]:


start_date = datetime.strptime('2018-03-14', '%Y-%m-%d')
end_date = datetime.strptime('2021-01-22', '%Y-%m-%d')

str_date_list = []

while start_date.strftime('%Y-%m-%d') != end_date.strftime('%Y-%m-%d'):
    str_date_list.append(start_date.strftime('%Y-%m-%d'))
    start_date += timedelta(days=1)

whole = pd.DataFrame(str_date_list, columns=['date'])
whole


# In[11]:


bitcoin = pd.read_csv('/Users/cline_local/python_develop/Analysis/bitcoin_price.csv')
bitcoin.columns = ['datetimes', 'price']
# bitcoin = bitcoin.sort_values(by='datetimes', ascending=False)
bitcoin


# ### 데이터 형태 맞춰주기

# In[12]:


corpus.columns = ['datetimes', 'title', 'article']
corpus


# In[13]:


merged = pd.merge(corpus, bitcoin, how='left')
merged = merged.fillna(method='ffill')
merged


# In[43]:


merged = merged[['datetimes', 'title', 'article', 'price']]
merged = merged.dropna(axis=0)
merged['price'] = merged['price'].astype(int)
merged


# In[15]:


# merged.to_csv('merged.csv')


# ---

# ### 상승률 및 하락률 만들어주기  
# (하루 단위로 할지? 일주일로 할지? 한달로 할지?)  
# 우선 하루 단위로 진행

# In[16]:


bitcoin_a = bitcoin.loc[:][:][:-1]
bitcoin_b = bitcoin.loc[:][:][1:].rename({'date':'1일 전 date', 'price':'1일 전 price'}, axis='columns').reset_index(drop=True)


# In[95]:


total_price = pd.merge(bitcoin_a, bitcoin_b, left_index=True, right_index=True)

total_price['price'] = total_price['price'].astype(int)
total_price['1일 전 price'] = total_price['1일 전 price'].astype(int)
total_price.columns = ['day-1', 'price-1', 'day', 'price']
total_price


# In[88]:


# total_price.to_csv('total_price.csv')


# In[18]:


total_price['labeling'] = np.where(total_price['price'] > total_price['price-1'], 'up',
                                  np.where(total_price['price'] == total_price['price-1'], '-',
                                          'down'))


# In[19]:


datelabel = ['price', 'labeling']
price_labeling = total_price[datelabel]
price_labeling


# In[45]:


dataframe = pd.merge(merged, price_labeling, how='left')
dataframe.dropna(axis=0)


# In[46]:


# dataframe.to_csv('dataframe.csv')


# In[20]:


import matplotlib.pyplot as plt

plt.figure(figsize=(25, 8))
plt.plot(price_labeling['price'])
plt.xlabel('date', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()


# In[89]:


# price_labeling.to_csv('price_labeling.csv')


# ---

# ### 데이터 전처리(Tokenize) / Corpus 구축(N-gram, Stemming, Lemmatization)
# Mecab or Khaiii?

# In[286]:


# Sentimental Analysis를 한다면 data imbalance문제는 없음.

dataframe['labeling'].value_counts()


# In[55]:


dataframe = dataframe.dropna(axis=0)
dataframe


# In[347]:


# khaiii 반복문을 통한 형태소 분석 불가
# Mecab만 사용하지 않고, ekonlpy.tag MPCK 분석기 사용(필요시 Mecab도 사용)
from khaiii import KhaiiiApi
api = KhaiiiApi()

def n_tokenize(data_article):
    morphs = []
    
    for sentence in tqdm(data_article['article']):
        for word in api.analyze(sentence):
            for morph in word.morphs:
                morphs.append((morph.lex, morph.tag))


# In[230]:


# !bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)


# ---
# # Check Point(read_csv)

# In[9]:


total_dataframe = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/price_article.csv')
total_dataframe = total_dataframe[['datetimes', 'title', 'article', 'price']]
total_dataframe


# In[10]:


# total_dataframe을 source 사용

date_article = total_dataframe[['datetimes', 'article']]
date_article['article'] = date_article['article'].astype(str)
date_article


# ---
# ##### 현재 Mac단말기에서 Mecab 구동이 잘 안되므로, Colab에서 형태소분석을 실행 후 corpus.csv 형태로 붙여준다.
# ---
# filename : eKoNLPy.tag_sentimental_mpck

# In[6]:


pos_tagger = []
pos_ngram_tagger = []
pos_score = []

for sentence in tqdm(date_article['article']):
    tokens = mpck.tokenize(sentence)
    if tokens != []:
    ngrams = mpck.ngramize(tokens)
    score = mpck.classify(tokens + ngrams, intensity_cutoff=1.3) # hyperparameters 설정이 필요함. 어조분류 score <- 추후, 결과 및 논의를 통해서 수정 요망(cline)
    
    pos_tagger.append(tokens)
    pos_ngram_tagger.append(ngrams)
    pos_score.append(score)
    
print(len(pos_tagger), len(pos_ngram_tagger), len(pos_score))


# In[40]:


pos_tagger[:1][0][0]


# ---

# ### 형태소 분석 후 사용할 품사 지정 : 명사 / 대명사 / 동사 / 형용사 / 부사

# In[7]:


df_pos_tagger = pd.DataFrame(pos_tagger)
df_pos_ngram_tagger = pd.DataFrame(pos_ngram_tagger)
df_pos_score = pd.DataFrame(pos_score)


# In[33]:


df = pd.concat([df_pos_tagger, df_pos_ngram_tagger, df_pos_score], axis=1)
df


# In[42]:


final = pd.concat([total_dataframe, df], axis=1)
final


# ---

# 단어별로 column이 구분되어지는게 아니라 하나의 기사에 여러단어가 들어가게끔 indexing 작업이 필요함(시각화 및 handling용이)  
# 함수 작성

# 또한, eKoNLPy의 intensity 및 pos/neg score는 기존 금융관련 형태소 분석을 바탕으로 만든 Polarity이기 떄문에, 비트코인 및 가상화폐관련 기사에는 부정적으로 반응할 수 밖에 없다고 보임  
# 따라서 새로운 Tone 및 polarity를 구성해야하는 필요성 느낌

# ---

# In[11]:


def n_tokenize(data):
    data['token'] = data['article'].apply(lambda x: x.split('.'))
    
    for idx, contents in tqdm(enumerate(data['token'])):
        only_token = []
        
        for sentence in contents:
            tokens = mpck.tokenize(sentence)
            
            if tokens != []:
                only_token.append(tokens)
            data['token'][idx] = only_token


# In[12]:


date_article


# In[13]:


n_tokenize(date_article)


# In[14]:


date_article


# ### corpus structure

# In[15]:


# bitcoin 가격 import

df_ti = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/total_price.csv')
df_ti = df_ti[['day-1', 'price-1', 'labeling']]
df_ti.columns = ['datetimes', 'price', 'Labeling']


# In[16]:


df_ti


# In[17]:


def corpus_maker(data):
    corpus = defaultdict(lambda : [0,0,0])
    
    for idx, token_book in tqdm(enumerate(data['token'])):
        for token_list in token_book:
            ngrams = mpck.ngramize(token_list)
            
            for ngram in ngrams:
                if data.iloc[idx].Labeling:
                    corpus[ngram][0] += 1
                    corpus[ngram][1] += 1
                    
                elif data.iloc[idx].Labeling == 'down':
                    corpus[ngram][0] += 1
                    corpus[ngram][2] += 1
                
                else:
                    corpus[ngram][0] += 1
                    
            for tokens in token_list:
                if tokens[-3:] != "XSV":
                    if data.iloc[idx].Labeling == 'up':
                        corpus[tokens][0] += 1
                        corpus[tokens][1] += 1
                        
                    elif data.iloc[idx].Labeling == 'down':
                        corpus[tokens][0] += 1
                        corpus[tokens][2] += 1
                        
                    else:
                        corpus[tokens][0] += 1
                        
    return corpus


# In[18]:


semi_df = date_article.merge(df_ti, on='datetimes')
semi_corpus_df = corpus_maker(semi_df)


# In[19]:


corpus_df = pd.DataFrame(semi_corpus_df).T
corpus_df.columns = ['total', 'up', 'down']
corpus_df


# In[20]:


# corpus_df.to_csv('corpus_df.csv')


# In[21]:


# method

from IPython.display import Image

corpus_df['polar_score'] = ((corpus_df['up'] / corpus_df['total']) / (corpus_df['up'].sum() / corpus_df['total'].sum())) / ((corpus_df['down'] / corpus_df['total']) / (corpus_df['down'].sum() / corpus_df['total'].sum()))
Image('/Users/cline_local/Datateam_cline/Analysis/How_to_calculate_polar_score.png')


# In[22]:


corpus_df


# In[26]:


corpus_df['polarity'] = ""


# In[27]:


for idx, value in tqdm(enumerate(corpus_df['polar_score'])):
    if corpus_df.iloc[idx].total > 14:
        if value > (13/10):
            corpus_df['polarity'][idx] = 'Hawkish'
        
        elif value < (10/13):
            corpus_df['polarity'][idx] = 'Dovish'
            
        else:
            corpus_df['polarity'][idx] = 'Nothing'
    
    else:
        corpus_df['polarity'][idx] = 'Nothing'


# In[23]:


corpus_df = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/corpus_df.csv')
corpus_df.columns = ['word', 'total', 'up', 'down']
corpus_df


# In[28]:


hawkish_dic = corpus_df[corpus_df['polarity'] == 'Hawkish']
dovish_dic = corpus_df[corpus_df['polarity'] == 'Dovish']


# In[29]:


hawkish_dic


# In[30]:


dovish_dic


# In[29]:


hawkish_dic.columns = ['word', 'total', 'up', 'down', 'polar_score', 'polarity']
hawkish_dic


# In[108]:


dovish_dic.columns = ['word', 'total', 'up', 'down', 'polar_score', 'polarity']
dovish_dic


# In[112]:


# dovish_dic.to_csv('dovish_dic.csv')


# In[113]:


# hawkish_dic.to_csv('hawkish_dic.csv')


# In[33]:


hawkish_dic.sort_values(by='up', ascending=False)


# In[44]:


# dovish_dic.sort_values(by='polar_score', ascending=False)
dovish_dic.sort_values(by='down', ascending=False)


# In[124]:


d = dovish_dic[['down']]
h = hawkish_dic[['up']]


# ### WordCloud 시각화

# In[176]:


from wordcloud import WordCloud

wc = WordCloud(font_path='/Users/cline_local/Datateam_cline/Analysis/godoFont/GodoB.otf',
              background_color='white',
              width=1000,
              height=1000,
              max_words=600,
              max_font_size=200)


# In[174]:


h_T = h.T
h_T = h_T.reset_index(drop=True)
h_T = h_T.to_dict('records')
h_T = h_T[0]
h_T


# In[177]:


wc.generate_from_frequencies(h_T)
wc.to_file('hawkish.png')


# In[179]:


d_T = d.T
d_T = d_T.reset_index(drop=True)
d_T = d_T.to_dict('records')
d_T = d_T[0]
d_T


# In[180]:


wc.generate_from_frequencies(d_T)
wc.to_file('dovish.png')


# In[181]:


Image('/Users/cline_local/Datateam_cline/Analysis/hawkish.png')


# In[182]:


Image('/Users/cline_local/Datateam_cline/Analysis/dovish.png')


# In[129]:


# Dovish = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/dovish_dic.csv')
# Hawkish = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/hawkish_dic.csv')


# In[130]:


Hawkish.columns = ['hawkish', 'total', 'up', 'down', 'polar_score', 'polarity']
Dovish.columns = ['dovish', 'total', 'up', 'down', 'polar_score', 'polarity']
hawkish = Hawkish[['hawkish']]
dovish = Dovish[['dovish']]


# In[131]:


hawkish_list = list(hawkish['hawkish'])
dovish_list = list(dovish['dovish'])


# In[132]:


len(hawkish_list), len(dovish_list)


# ---

# ### External Data(검증)

# In[3]:


# main method
# 1.22 ~ 2.7일까지의 기사를 External Data(test datasets)로 사용

news_title = []
date_list = []
news_article = []

for i in tqdm(range(1, 7)):
    url = 'https://www.coindeskkorea.com/news/articleList.html?page={}&total=6048&box_idxno=&view_type=sm'.format(i)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    
    for j in range(1, 121):
        title = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-titles a strong'.format(j))
        date = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block div.list-dated'.format(j))
        article = soup.select('#user-container div.float-center.custom-m.mobile.template.list.max-width-1250 div.user-content section article div.article-list section div:nth-of-type({}) div.text-block p a'.format(j))
        
        news_title.append(title)
        date_list.append(date)
        news_article.append(article)
        
print("총 {}개의 전처리가 필요한 제목".format(len(news_title)))
print("총 {}개의 전처리가 필요한 날짜".format(len(date_list)))
print("총 {}개의 전처리가 필요한 기사".format(len(news_article)))


# In[17]:


test_title = []

for title in tqdm(news_title):
    for tit in title:
        for t in tit:
#             print(t)
            test_title.append(t)
            
print("전처리 후 제목: {}".format(len(test_title)))
test_title[:10]


# In[23]:


test_source = []

for news in tqdm(news_article):
    for new in news:
        for nw in new:
            test_source.append(nw)
            
print(len(test_source))
print(test_source[:5])


test_clear_article = []

for i in tqdm(test_source):
    test_clear_article.append(i[6:-5])

print("전처리 후 제목: {}".format(len(test_clear_article)))
test_clear_article[:5]


# In[24]:


normal_test_date = []

for date_ in tqdm(date_list):
    for date in date_:
        for d in date:
            normal_test_date.append(d)
            
normal_test_date = normal_test_date[1::2]
print("전처리 후 제목: {}".format(len(normal_test_date)))
normal_test_date[:10]


# In[25]:


clear_test_date = []

for _date in normal_test_date:
    clear_test_date.append(_date[1:-6])
    
clear_test_date[:5]


# In[29]:


test_day = pd.DataFrame(clear_test_date)
test_day.columns = ['datetimes']
test_article = pd.DataFrame(test_clear_article)
test_article.columns = ['article']


# In[32]:


test_dataframe = pd.concat([test_day, test_article], axis = 1)
test_dataframe


# In[6]:


test_date = pd.read_clipboard()
test_date.columns = ['datetimes', 'price']
test_date


# In[7]:


# test_date.to_csv('test_data.csv')


# In[11]:


test_a = test_date.loc[:][:][:-1]
test_b = test_date.loc[:][:][1:].rename({'date':'1일 전 date', 'price':'1일 전 price'}, axis='columns').reset_index(drop=True)


# In[12]:


pd_test = pd.merge(test_a, test_b, left_index=True, right_index=True)

pd_test['price'] = pd_test['price'].astype(int)
pd_test['1일 전 price'] = pd_test['1일 전 price'].astype(int)
pd_test.columns = ['day-1', 'price-1', 'day', 'price']
pd_test


# In[13]:


pd_test['labeling'] = np.where(pd_test['price'] > pd_test['price-1'], 'up',
                                  np.where(pd_test['price'] == pd_test['price-1'], '-',
                                          'down'))


# In[33]:


datelabel = ['day', 'price', 'labeling']
test_label = pd_test[datelabel]
test_label.columns = ['datetimes', 'price', 'labeling']
test_label


# In[34]:


total_test = pd.merge(test_dataframe, test_label, on='datetimes')
total_test


# In[35]:


# total_test.to_csv('total_test_dataframe.csv')


# In[184]:


total_test = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/total_test_dataframe.csv')
total_test[['datetimes', 'article', 'price']]
total_test


# In[185]:


n_tokenize(total_test)
total_test


# In[186]:


total_test['article'] = total_test['article'].astype(str)


# In[187]:


doc_score_list = []


for i in range(len(total_test)):
    news_score_list = []
    
    for sentence in total_test['token'][i]:
        dict_sentence = [[], []]
        
        for token in sentence:
            if token in dovish_list:
                dict_sentence[1].append(token)
            elif token in hawkish_list:
                dict_sentence[0].append(token)
            
            tone_score = []
            if (len(dict_sentence[0]) + len(dict_sentence[1])) == 0:
                tone_score.append('')
            else:
                tone_score.append((len(dict_sentence[0]) - len(dict_sentence[1])) / (len(dict_sentence[0]) + len(dict_sentence[1])))
            
            news_score_list.append(tone_score)
            
        doc_score_list.append(news_score_list)
        
doc_score_list[:5]


# In[188]:


each_doc_tone = []

for i in range(len(doc_score_list)):
    doc_tone = []
    
    if doc_score_list[i] == []:
        doc_tone.append('')
    
    else:
        no_of_hawkish = 0
        no_of_dovish = 0
        
        for j in range(len(doc_score_list[i])):
            if doc_score_list[i][j][0] != '':
                if doc_score_list[i][j][0] > 0:
                    no_of_hawkish += 1
                elif doc_score_list[i][j][0] < 0:
                    no_of_dovish += 1
            else:
                continue
        
        tone = (no_of_hawkish - no_of_dovish) / (no_of_hawkish + no_of_dovish)
        doc_tone.append(tone)
        
    each_doc_tone.append(doc_tone)


# In[189]:


total_test


# In[190]:


total_test['tone'] = total_test['datetimes']

for i in tqdm(range(len(each_doc_tone))):
    total_test['tone'][i] = each_doc_tone[i][0]


# In[191]:


total_df = total_test[['datetimes', 'price', 'tone', 'labeling']]
total_df


# In[192]:


int_tone = total_df['tone'].tolist()
int_price = total_df['price'].tolist()

print(int_tone[:5])
print(int_price[:5])


# In[36]:


# total_df.to_csv('result.csv')


# In[193]:


total_df['tone'] = total_df['tone'].astype('float64')
total_df['price'] = total_df['price'].astype('float64')


# In[194]:


grip = total_df[['price', 'tone', 'labeling']].groupby(total_df['datetimes'])
grip = grip.mean()
grip


# In[205]:


# grip.to_csv('grip.csv')


# In[197]:


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.title('original_comparison', fontsize=20, pad=20)
plt.xlabel('time')
plt.ylabel('price_tone')
plt.plot(grip)
plt.legend(['price', 'tone_score'])
plt.show()


# In[198]:


grip['price'].corr(grip['tone'])


# In[207]:


f = pd.read_csv('/Users/cline_local/Datateam_cline/Analysis/grip.csv')
f


# In[208]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minmax = MinMaxScaler()
std = StandardScaler()

x_minmax = minmax.fit_transform(grip)
x_std = std.fit_transform(grip)


# In[212]:


x_std_df = pd.DataFrame(x_std)
x_std_df.columns = ['std_price', 'std_tone']
x_std_df2 = x_std_df.set_index(f['datetimes'])
x_std_df2


# In[215]:


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.title('std_graph', fontsize=20, pad=20)
plt.xlabel('time')
plt.ylabel('standardscaler')
plt.plot(x_std_df2)
plt.legend(['std_price', 'std_tone'])
plt.show()


# In[82]:


x_std_df['std_price'].corr(x_std_df['std_tone']) * 100


# In[78]:


x_minmax_df = pd.DataFrame(x_minmax)
x_minmax_df.columns = ['minmax_price', 'minmax_tone']
x_minmax_df


# In[89]:


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.title('MinMax_graph', fontsize=20, pad=20)
plt.xlabel('time')
plt.ylabel('standardscaler')
plt.plot(x_minmax_df)
plt.legend(['minmax_price', 'minmax_tone'])
plt.show()


# In[121]:


x_minmax_df['minmax_price'].corr(x_minmax_df['minmax_tone']) * 100


# ### 추가검증 요청

# 테슬라, 1.6조원 비트코인 매수…‘전기차 결제도 비트코인으로’  
# https://www.coindeskkorea.com/news/articleView.html?idxno=72735  
# 캐나다 금융당국, 비트코인 ETF 첫 승인  
# http://www.coindeskkorea.com/news/articleView.html?idxno=72761  
# 모건스탠리, 비트코인 투자 검토 중  
# http://www.coindeskkorea.com/news/articleView.html?idxno=72762  
# 미국 SEC-리플, 합의 가능성 희박...공동서한 법원 제출  
# http://www.coindeskkorea.com/news/articleView.html?idxno=72778

# In[8]:


date_list = ['2021-02-09']
news = ['''1. 테슬라가 15억달러 어치의 비트코인 매수. 글로벌 전기차 기업인 테슬라가 15억달러 어치의  비트코인을 매수했다고 밝혔다. 미국 증권거래위원회(SEC) 연간보고서에 따르면 테슬라는 지난 1월 기업자금을 “더 유연하고 효율적으로 운영하기 위해 투자 정책을 업데이트했으며, 이 정책에 따라 총 15억달러 어치의 비트코인을 취득했다”고 밝혔다. 작년 말 기준 테슬라 현금과 현금등가물은 총 190억달러 정도다.
테슬라의 비트코인 매수 소식이 전해지자 개당 3만9000달러 선을 횡보하던 비트코인 가격은 40여분만에 글로벌 기준 4만5000달러 가깝게 급등했다. 가격 급등으로 암호화폐 거래소 사용량이 늘어나면서, 바이낸스 등 일부 글로벌 거래소는 몇십분 간 사용이 불가능한 상태가 되기도 했다. 테슬라는 비트코인을 “가까운 미래에 결제수단으로 용인할 예정”이라고도 밝혔다.
테슬라 CEO인 일론 머스크는 최근 음성 소셜미디어 애플리케이션 클럽하우스(Clubhouse)에서 "8년 전에 비트코인을 샀어야 했다"며 "비트코인을 현재 긍정적으로 보고 있으며 지지한다"고 말한 바 있다. 머스크를 포함한 많은 기업인이 코로나19로 인한 경제적 혼란 속에서 비트코인 등 암호화폐에 관심을 보이고 있다.
'헤지펀드 전설'로 불리는 빌 밀러도 며칠 전 BTC(비트코인) 노출 비중을 15%까지 높이기 위해 그레이스케일 비트코인 신탁(GBTC)를 매입하겠다고 밝혔다.
6일(현지시간) SEC에 제출한 규제 신고서에 따르면, 밀러 밸류 펀드는 "그레이스케일 비트코인 신탁 상품에 투자해 간접적으로 비트코인에 대한 노출 방안을 모색할 수 있다"며 "그 결과 비트코인에의 노출 비중이 15%를 넘기면 추가 매입은 없다"고 밝혔다.(그레이스케일은 코인데스크의 자매회사다.)
빌 밀러는 자사 최상위 펀드인 '밀러 오퍼튜니티 신탁'을 통해 이번 투자를 진행한다는 방침인데 밀러는 지난해 12월31일 기준 22억5000만달러의 자금을 운용하고 있으며, 이 중 최대 3억3700만달러(3800억원)를 GBTC에 투자할 계획이다.''']


# In[9]:


news_article = []

for i in tqdm(news):
    news_article.append(i)
    
news_article


# In[14]:


test1 = pd.DataFrame(news_article)
test1.columns = ['article']

n_tokenize(test1)


# In[19]:


test1


# In[45]:


doc_score_list = []


for i in range(len(test2)):
    news_score_list = []
    
    for sentence in test2['token'][i]:
        dict_sentence = [[], []]
        
        for token in sentence:
            if token in dovish_list:
                dict_sentence[1].append(token)
            elif token in hawkish_list:
                dict_sentence[0].append(token)
            
            tone_score = []
            if (len(dict_sentence[0]) + len(dict_sentence[1])) == 0:
                tone_score.append('')
            else:
                tone_score.append((len(dict_sentence[0]) - len(dict_sentence[1])) / (len(dict_sentence[0]) + len(dict_sentence[1])))
            
            news_score_list.append(tone_score)
            
        doc_score_list.append(news_score_list)
        
doc_score_list[:5]


# In[46]:


each_doc_tone = []

for i in range(len(doc_score_list)):
    doc_tone = []
    
    if doc_score_list[i] == []:
        doc_tone.append('')
    
    else:
        no_of_hawkish = 0
        no_of_dovish = 0
        
        for j in range(len(doc_score_list[i])):
            if doc_score_list[i][j][0] != '':
                if doc_score_list[i][j][0] > 0:
                    no_of_hawkish += 1
                elif doc_score_list[i][j][0] < 0:
                    no_of_dovish += 1
            else:
                continue
        
        tone = (no_of_hawkish - no_of_dovish) / (no_of_hawkish + no_of_dovish)
        doc_tone.append(tone)
        
    each_doc_tone.append(doc_tone)


# In[47]:


test2


# In[26]:


test1['tone'] = test1['article']

for i in tqdm(range(len(each_doc_tone))):
    test1['tone'][i] = each_doc_tone[i][0]


# In[30]:


int_tone = test1['tone'].tolist()

print(int_tone[:5])


# In[37]:


news2 = ['''캐나다 금융 당국이 북미 최초로 비트코인 상장지수펀드(ETF) 출시를 승인했다.
온타리오증권위원회(OSC)는 캐나다 기반 자산운용사 퍼포즈 인베스트먼트가 신청한 비트코인 ETF 출시를 11일 승인했다. 
캐나다 투자 관리 회사 3iQ가 상장한 펀드를 비롯한 복수의 폐쇄형 (close-ended) 비트코인 펀드가 토론토 증권거래소에 이미 상장돼 있지만, 이는 ETF와는 차이가 있다. ETF의 경우 증권을 지속해서 발행하는 반면, 폐쇄형 펀드는 최초 판매 시와 판매 재개 시에만 증권을 발행한다. 
퍼포즈 인베스트먼트가 온라인에 게시한 설명 자료에 따르면, 이번에 승인을 얻은 펀드는 비트코인 가격 변동성에서 수수료와 비용을 제외한 만큼의 실적을 추종한다. 퍼포즈 인베스트먼트는 이 ETF가 비트코인 가격의 단기적 변동성에 투기하진 않을 것이라고 설명했다. 
퍼포즈 인베스트먼트는 비트코인 ETF가 장기적 관점의 자산 성장과 매력적인 위험 조정 수익률을 추구하며 '고위험'을 용인할 수 있는 투자자를 주된 대상으로 한다고 밝혔다. 
일반적으로 ETF는 바스켓 내 복수의 상품에 투자한다는 점에서 뮤추얼 펀드와 유사하다. 다만, 거래소를 통해 거래된다는 점에선 증권에 더 가깝다. ETF 상품의 위험도는 어떤 자산을 기초 자산으로 하느냐에 따라 달라진다. 비트코인 ETF는 대개 "고위험" 상품으로 여겨진다.
투자를 통해 "지속적인 수입원"을 만들고자 하는 이들에겐 일반적으로 비트코인 ETF 투자를 권하지 않는다. 
퍼포즈 인베스트먼트 비트코인 ETF의 수수료는 관리 수수료와 운영 및 거래 비용 등으로 구성된다. 현재 연 수수료는 비트코인 가치의 1%로 설정돼 있다. 펀드가 이제 막 출시된 만큼, 운영비와 거래 수수료는 아직 제공되지 않았다.
에릭 발추나스 블룸버그 시니어 애널리스트는 이번 소식이 미국이 제재를 가한 비트코인 ETF에 '좋은 신호'가 될 거라고 믿는다고 말했다.
그는 트위터를 통해 "캐나다의 진보적인 규제 당국을 사랑한다. 아마도 그들이 정상이고, 미국 증권거래위원회(SEC)가 너무 보수적인 것인지도 모른다"고 말했다. 그는 "어찌 됐건 미국도 곧 그 뒤를 따르게 될 것"이라고 덧붙였다.
토론토 증권거래소는 해당 펀드를 캐나다 달러로 상장할 것으로 보인다. 포트폴리오 및 펀드 관리는 퍼포즈 인베스트먼트가 담당한다.''',
        '''투자은행 모건스탠리에서 1500억달러 규모의 자금을 운용하는 투자전문 자회사 카운터포인트 글로벌(Counterpoint Global)을 통해 비트코인에 베팅하는 방안을 검토하고 있다고 블룸버그가 복수의 소식통을 인용해 보도했다.
즉각적인 투자가 가능한 상황은 아니다. 블룸버그는 카운터포인트 글로벌이 계획하고 있는 비트코인 투자를 추진하기 위해서는 모건스탠리와 규제당국의 허가를 먼저 받아야 한다고 설명했다.
모건스탠리가 업계 대표 암호화폐인 비트코인에 투자하는 것은 이번이 처음이 아니다. 코인데스크 보도에 따르면, 모건스탠리는 고비중 비트코인 투자로 최근 유명해진 나스닥 상장기업 마이크로스트래티지(MicroStrategy) 지분을 11% 가까이 보유하고 있다.
모건스탠리 애널리스트들은 비트코인이 미국 달러를 위협할 수 있는 잠재력을 갖고 있다고 지적하면서도, 최근 공개된 보고서에서는 비트코인을 장기보유하는 투자자가 많아질수록 결제 수단으로의 가치는 떨어질 수 있다고 강조했다.
카운터포인트 글로벌은 현재 20개 가까운 펀드를 운용하고 있으며, 블룸버그는 이 중 5개 펀드가 지난해 100% 이상의 수익률을 달성했다고 전했다. 모건스탠리는 블룸버그 보도 내용에 대한 언급을 삼가고 있다.''',
        '''미국 증권거래위원회(SEC)와 리플 간 합의 가능성이 희박한 것으로 나타났다.
SEC와 리플은 16일(현지시간) 아날리사 토레스(Analisa Torres) 뉴욕 남부지방법원 연방판사에게 제출한 서한을 통해 "현재로서는 (재판 전) 해결 가능성이 없다"고 밝혔다.
이어 "트럼프 행정부 시절 합의안에 대한 논의가 이어졌으나 당시 주축을 이뤘던 부서장들이 SEC를 떠났다"며 "새로운 합의가 이뤄질 경우 즉시 법원에 통보하겠다"고 덧붙였다.    
SEC와 리플은 증거개시(Discovery: 소송 당사자들이 재판에 앞서 증거와 정보를 서로 공개하게 하는 제도)를 8월 16일까지 종료하기로 동의했다. 
SEC는 이번 서한에서 "리플과 크리스 라센 의장은 XRP가 ‘투자 계약’으로 간주될 위험이 있으며 증권법상 증권으로 간주될 수 있다고 경고하는 법률 메모를 받은 바 있다"며 리플에게 추가 증언 녹취록을 제출할 것을 요구했다. 하지만 리플은 "SEC의 요청은 부적절하고 법적 근거도 부족하다"며 이를 거절했다. 
앞서 지난해 12월 SEC는 XRP가 미등록 증권이라는 이유로 리플을 증권법 위반 혐의로 고소했다. SEC와 리플 간 최초 변론 전 회의(pretrial conference)는 2월 22일로 예정됐다.''']


# In[40]:


news_article = []

for i in news2:
    news_article.append(i)
    
len(news_article)


# In[42]:


test2 = pd.DataFrame(news_article)
test2.columns = ['article']
test2


# In[43]:


n_tokenize(test2)
test2


# In[48]:


test2['tone'] = test2['article']

for i in tqdm(range(len(each_doc_tone))):
    test2['tone'][i] = each_doc_tone[i][0]


# In[50]:


int_tone = test2['tone'].tolist()

print(int_tone[:5])


# In[ ]:





# ---

# ## 상승/하락 단어를 Vetorize 후 price의 변동성을 LSTM Algorithm으로 학습  
#  - 빗, 썸, 체인, 베이스, 법, 설명 등등 상승/하락 단어들을 Vectorize
#  - 0.242345, 0.298523, 0.534539, 0.438434 등으로 Vertor화
#  - x를 역치행렬(T)으로 구성 -> ex.(0.242345, 0.298523, 0.534539, 0.438434, ..)
#  - y를 price 혹은 up/down으로 구성(연산 방식 혹은 1, 0의 sigmoid_binary 형식으로 target)
#  - 그 전에, 충분한 dictionary_corpus를 구축(코인데스크코리아를 제외한, 미국연준 및 네이버뉴스 등 공신력이 있는 기사 및 기관)
#  - Framework : 주 Pytorch / 예 Tensorflow LSTM 10 layers(Batch Normalize)_Attention Weights

# ### Corpus를 Word2Vec

# In[105]:


import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from nltk.tokenize import word_tokenize, sent_tokenize

