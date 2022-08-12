#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:57:18 2021

@author: tylervo
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import string
nltk.download("stopwords")
nltk.download("punkt")
#%%Save your data in a .csv file (or other format as appropriate for your data set and project scenario): 
un = pd.read_csv(r"/Users/tylervo/Desktop/UVA/Fall2021/DS2001/un-general-debates.csv")
un[["year", "country"]].groupby("year").count().plot(kind="bar")
#%% Perform data pre-processing, data cleaning, outlier removal, and so on to sanitize your data as necessary: 
#%% Data frame of UNSC Permenant Members (China, France, Russia, UK, US)
unsc_members = ['USA','GBR','FRA','RUS','CHN']
unsc = un[un.country.isin(unsc_members)]

#%% Text preparation, Defining Function  
unsc['text'] = unsc['text'].str.lower().map(lambda x: re.sub('\W+',' ', x))
formal_words = ['united nation', 'united nations', 'general assembly', 'republic of', 'secretary general', 'the world', 'international community', 'security council', 'member state', 'country', 'must', 'many']

def preprocess_text(text):
    for x in formal_words: #Remove formal words in speeches
        text = text.replace(x, '')
    tokens = nltk.word_tokenize(text)
    stop_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    tokens = [token for token in tokens if len(token) > 3 and token not in stop_words]
    return tokens

from collections import Counter

#%% Adding a collum 'tokens' with cleaned debate speech
unsc['tokens'] = unsc['text'].apply(preprocess_text)
print(unsc['tokens'][7])
unsc['token_freq'] = unsc['tokens'].apply(Counter)
unsc['sorted'] = unsc['token_freq'].apply(sorted[0:25])
#%%
topics = ['united states', 'peacekeeping', 'dictatorship', 'war', 'peace', 'nuclear weapons', 'terrorism', 'genocide', 'human rights', 'middle east', 'europe', 'asia', 'political', 'economic', 'women', 'refugee', 'security', 'china', 'democracy', 'crimes against humanity']
plt.bar(unsc['token_freq'][7].keys(), unsc['token_freq'][7].values())

#%% Create at least 2 visualizations that you find interesting/useful: 
for i, row in unsc.iterrows():
    sess = dict(nltk.FreqDist(row['tokens']))
    sort_sess = sorted(sess.items(), key=lambda x: x[1], reverse=True)
#%% Other statistical test: 

#%% Write at least two unit tests. For example, these might be short tests to show that two different functions work as intended: 
import unittest
from unittest import TestCase
tc = TestCase()
def test1_preprocess_text(self):
    #input
    text = "Hello, this is the general assembly, where we will be talking about politics and stuff."
    #expected
    expected1 = ['Hello', 'talking', 'politics', 'stuff'] #removes formal and unneccesary words
    #actual
    actual1 = preprocess_text(text)
    
    print(tc.assertEqual(actual1, expected1))
if __name__ == '__main__':
    unittest.main()
