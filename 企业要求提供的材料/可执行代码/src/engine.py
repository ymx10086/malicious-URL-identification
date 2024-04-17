import pandas as pd
import numpy as np
import os

DATADIR = "../dataset/train1.csv"
os.path.exists(DATADIR)

df = pd.read_csv(DATADIR)

import urllib,re
from urllib import request
import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup
from lxml import etree

words_ = []
LIMIT= 10000
df = df.iloc[:LIMIT]
ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240"
for _ in range(min(LIMIT,df.shape[0])):
    i = _ + LIMIT
    if _ % 100 == 0:
        print("{}-{} completed".format(_/100,df.shape[0]/100))
    url = 'http://'+df.url.iloc[i]
    try: 
        res = requests.request('GET',url,headers = {'User-agent':ua},allow_redirects=True)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text,'html.parser')
        text = soup.get_text()
        regex = re.compile("[\\n\s0-9\.：”“]")
        text = regex.sub('',text)
        words_.append(text)
    except:
        words_.append(None)


df['words'] = words_
df.to_csv("train_text2.csv")
# from snownlp import SnowNLP
# import thulac
# text = "sss"