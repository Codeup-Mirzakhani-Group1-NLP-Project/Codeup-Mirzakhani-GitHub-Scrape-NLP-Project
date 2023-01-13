###################
##### IMPORTS #####
###################

import unicodedata
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import re
import json
import nltk
import requests
import pandas as pd
import scipy.stats as stats
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Union, cast

## Plots, Graphs, & Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import DateFormatter

# ------------- #
# Local Imports #
# ------------- #

## importing sys
import sys

## adding 00_helper_files to the system path as First Location to look
sys.path.insert(0, '/Users/qmcbt/codeup-data-science/00_helper_files')
## adding 03_projects Personal Work folder for current project to the system path as Second Location to look
sys.path.insert(1, '/Users/qmcbt/codeup-data-science/03_projects/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project/justin_docs')
## adding 03_projects Root folder for current project to the system path as Third Location to look
sys.path.insert(2, '/Users/qmcbt/codeup-data-science/03_projects/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project')

## env containing sensitive access credentials
import env
from env import github_token, github_username

## Import Helper Modules
import acquire as ac
import prepare as pr
import modeling as m



#################################
##### Train, Validate, Test #####
#################################

df = pr.get_clean_df()
train, validate, test = pr.split_data(df, explore=True)
target = 'language'



#########################
##### Justin_GLOBAL #####
#########################

# split languages into seperate DataFrames
js_lang = train[train[target] == 'JavaScript']
cs_lang = train[train[target] == 'C#']
jv_lang = train[train[target] == 'Java']
py_lang = train[train[target] == 'Python']
    
# create word groups on lemmatized column 
js_lem = (' '.join(js_lang[js_lang[target] == 'JavaScript']['lemmatized'])).split()
cs_lem = (' '.join(cs_lang[cs_lang[target] == 'C#']['lemmatized'])).split()
jv_lem = (' '.join(jv_lang[jv_lang[target] == 'Java']['lemmatized'])).split()
py_lem = (' '.join(py_lang[py_lang[target] == 'Python']['lemmatized'])).split()
all_lem = (' '.join(df['lemmatized'])).split()
    
# create word groups on clean column
js_clean = (' '.join(js_lang[js_lang[target] == 'JavaScript']['clean'])).split()
cs_clean = (' '.join(cs_lang[cs_lang[target] == 'C#']['clean'])).split()
jv_clean = (' '.join(jv_lang[jv_lang[target] == 'Java']['clean'])).split()
py_clean = (' '.join(py_lang[py_lang[target] == 'Python']['clean'])).split()
all_clean = (' '.join(df['clean'])).split()
    
# Create frequency Series for lem
js_lem_freq = pd.Series(js_lem).value_counts()
cs_lem_freq = pd.Series(cs_lem).value_counts()
jv_lem_freq = pd.Series(jv_lem).value_counts()
py_lem_freq = pd.Series(py_lem).value_counts()
all_lem_freq = pd.Series(all_lem).value_counts()

# Display lemmatized word count frequency by language
lem_word_counts = pd.concat([js_lem_freq, cs_lem_freq, jv_lem_freq, py_lem_freq, all_lem_freq],
                             axis=1).fillna(0).astype(int)
lem_word_counts.columns = ['JavaScript','C#','Java', 'Python', 'All']

# Create Bi-Grams for each language
js_2_gram = list(nltk.ngrams(js_lem, 2))
cs_2_gram = list(nltk.ngrams(cs_lem, 2))
jv_2_gram = list(nltk.ngrams(jv_lem, 2))
py_2_gram = list(nltk.ngrams(py_lem, 2))
all_2_gram = list(nltk.ngrams(all_lem, 2))

# Create N-Grams of 3N for each language
js_3_gram = list(nltk.ngrams(js_lem, 3))
cs_3_gram = list(nltk.ngrams(cs_lem, 3))
jv_3_gram = list(nltk.ngrams(jv_lem, 3))
py_3_gram = list(nltk.ngrams(py_lem, 3))
all_3_gram = list(nltk.ngrams(all_lem, 3))

# Create Variables for lemmatized length and sentiment for each language

# JavaScript
js_lem_length = train[train[target] == 'JavaScript'].lem_length
js_sentiment = train[train[target] == 'JavaScript'].sentiment
# C#
cs_lem_length = train[train[target] == 'C#'].lem_length
cs_sentiment = train[train[target] == 'C#'].sentiment
# C#
jv_lem_length = train[train[target] == 'Java'].lem_length
jv_sentiment = train[train[target] == 'Java'].sentiment
# C#
py_lem_length = train[train[target] == 'Python'].lem_length
py_sentiment = train[train[target] == 'Python'].sentiment
# All
all_lem_length = train.lem_length
all_sentiment = train.sentiment



#########################
##### Justin_GLOBAL #####
#########################

def qmcbt_viz_01():

    # sort by one language
    return lem_word_counts.sort_values(['JavaScript'], ascending=False)[1:2]
    
def qmcbt_viz_02():
    # show highest over 'All' word count campared by language
    plt.rc('font', size=18)
    lem_word_counts.sort_values('JavaScript', 
                                ascending=False)[['JavaScript',
                                                'C#',
                                                'Java', 
                                                'Python']][1:2].plot.barh()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt.show()

def qmcbt_viz_03():
    # sort by one language
    return lem_word_counts.sort_values(['Java'], ascending=False).head(1)

def qmcbt_viz_04():
    # show highest over 'All' word count campared by language
    plt.rc('font', size=18)
    lem_word_counts.sort_values('Java', 
                                ascending=False)[['JavaScript',
                                                'C#',
                                                'Java', 
                                                'Python']][0:1].plot.barh()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt.show()

def qmcbt_viz_05():
    # Display top Bi-Gram pair for JavaScript lemmatized
    pd.Series(js_2_gram).value_counts().head(5).plot.barh()
    plt.title('JavaScript Bi-Grams')
    return plt.show()

def qmcbt_viz_06():
    # is the distribution for sentiment different for any of the languages

    # setting basic style parameters for matplotlib
    plt.figsize=(13, 7)
    plt.style.use('seaborn-darkgrid')

    # KDE Plot
    sns.kdeplot(js_sentiment, label = 'JavaScript')
    sns.kdeplot(cs_sentiment, label = 'C#')
    sns.kdeplot(jv_sentiment, label = 'Java')
    sns.kdeplot(py_sentiment, label = 'Python')
    plt.legend(['JavaScript', 'C#', 'Java', 'Python'])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    return plt.show()

def qmcbt_stat_01():
    #import scipy.stats as stats
    
    # Set alpha
    alpha = α = 0.05

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    f_val, p_val = stats.f_oneway(js_sentiment, cs_sentiment, jv_sentiment, py_sentiment)

    print(f'f_val: {f_val}')
    print(f'p_val: {p_val}')
    print('------------------------------')

    if p_val < α:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')

