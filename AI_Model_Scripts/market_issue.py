import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
import nltk
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from pandas.core.frame import DataFrame

## crawling
def get_attractions(url, data=None):
    data_sume=[]
    wb_data=requests.get(url)
    soup=BeautifulSoup(wb_data.text,'html.parser')
    titles=soup.select('.card-content > a > h4.headline')
    times = soup.select('.fixed-bottom >  span.time ')
    for title, time in zip (titles, times):
        data={
            'title':title.get_text(),
            'time':time.get_text(),
        }
        data_sume.append(data)
    return data_sume

urls=['https://bitcoinmagazine.com/sections/regulation/{}/'.format(str(i)) for i in range (1,6)]
data_sum=[]
for single_url in urls:
    data_sum.append(get_attractions(single_url))
#####


print(len(data_sum))



title=[]
for i in range(5):
    for j in range(19):
      title.append(data_sum[i][j]['title'])
print(title)
print(len(title))
time=[]
for i in range(5):
    for j in range(19):
     time.append(data_sum[i][j]['time'])
print(len(time))

c={
    "time": time,
    "title": title
}
print(c)

### NTLK　패키니를 이용하여 자연어 처리, 제목 분석
df_stocks=DataFrame(c)
df=df_stocks[['time']].copy()
df["compound"] = ''# compound sentence tendency judgement
df["neg"] = ''#negative sentence
df["neu"] = ''#neutral sentence
df["pos"] = ''#positive sentence

## conducting the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
for date, row in df_stocks.T.iteritems():
    try:
        sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'title'])
        ss = sid.polarity_scores(sentence)
        df.at[date, 'compound'] = ss['compound']
        df.at[date, 'neg'] = ss['neg']
        df.at[date, 'neu'] = ss['neu']
        df.at[date, 'pos'] = ss['pos']
    except TypeError:
        print(df_stocks.loc[date, 'title'])
        print(date)
print(df)

df.to_csv ("Market_Issue.csv" )


#### SentimentIntensityAnalyzer test 
view = [ "ICO 신규 규제안 발표",
        "Loved the ambience",
        "The place is not easy to locate"]
sid = SentimentIntensityAnalyzer()
for sen in view:
    print(sen)
    ss = sid.polarity_scores(sen)
    for k in ss:
        print('{0}:{1},'.format(k, ss[k]), end='')
    print()
####
