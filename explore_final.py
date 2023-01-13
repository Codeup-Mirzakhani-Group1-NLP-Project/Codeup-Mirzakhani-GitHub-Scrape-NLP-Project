import os
import json
import unicodedata
import re
import prepare as pr
import acquire as ac
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from env import github_token, github_username
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pr.get_clean_df()
train, validate, test = pr.split_data(df, explore=True)
target = 'language'
seed = 42

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
all_lem = (' '.join(train['lemmatized'])).split()
    
# create word groups on clean column
js_clean = (' '.join(js_lang[js_lang[target] == 'JavaScript']['clean'])).split()
cs_clean = (' '.join(cs_lang[cs_lang[target] == 'C#']['clean'])).split()
jv_clean = (' '.join(jv_lang[jv_lang[target] == 'Java']['clean'])).split()
py_clean = (' '.join(py_lang[py_lang[target] == 'Python']['clean'])).split()
all_clean = (' '.join(train['clean'])).split()

cs_freq_clean = pd.Series(cs_clean).value_counts()
js_freq_clean = pd.Series(js_clean).value_counts()
jv_freq_clean = pd.Series(jv_clean).value_counts()
py_freq_clean = pd.Series(py_clean).value_counts()
all_freq_clean = pd.Series(all_clean).value_counts()

# python string column created
js_lang['lem str len'] = js_lang['lemmatized'].str.len()
jv_lang['lem str len'] = jv_lang['lemmatized'].str.len()
cs_lang['lem str len'] = cs_lang['lemmatized'].str.len()
py_lang['lem str len'] = py_lang['lemmatized'].str.len()

# length of strings 
js_lem_length = js_lang['lem str len']
jv_lem_length = jv_lang['lem str len']
cs_lem_length = cs_lang['lem str len']
py_lem_length = py_lang['lem str len']


########ALLANTE'S EXPORE CODE######################
def data_representation(df):
    '''this function will create a data frame that shows the count and percentage of the target variable'''
    # creating dataframe of languages and their count and percentages 
    languages_table = pd.concat([df.language.value_counts(),round(df.language.value_counts(normalize=True),2)], axis=1)
    # creating column names for dataframe
    languages_table.columns = ['count', 'percent']
    # displaying table of information 
    return languages_table


def pie(df):
    '''this function will create a pie chart of the target variable and table '''
    #adjusts size of 
    plt.figure(figsize=(20,10))
    #define data
    data = [35,32,22,11]
    labels = ['JavaScript', 'Python', 'C#', 'Java']

    #define Seaborn color palette to use
    colors = sns.color_palette('pastel')[0:5]

    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.show()


def sort_by_language(variable,cs_freq_clean,js_freq_clean, jv_freq_clean, py_freq_clean,all_freq_clean):
    '''sorts table by most frequently used word based on column name selected'''
     # dataframe of word counts 
    clean_wordcount = (pd.concat([cs_freq_clean,js_freq_clean, jv_freq_clean, py_freq_clean,all_freq_clean], axis=1, sort=True)
                .set_axis(['c_sharp','javascript', 'java', 'python','all'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    # filtering most words by column
    return clean_wordcount.sort_values(by=variable, ascending=False).head(10)


def csharp_bigrams_lem(cs_lem):
    '''this function will create bar chart that will display top 10 bi grams'''
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_lem_bigrams = (pd.Series(nltk.ngrams(cs_lem, 2)).value_counts().head(10))
    # sorts 
    top_10_csharp_lem_bigrams.sort_values(ascending=True).plot.barh(color='green', width=.9, figsize=(10, 6))

    plt.title('10 Most frequently occuring c# bigrams (Lemmatized)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_lem_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)


def csharp_bigrams_clean(cs_lem):
    '''this function will create bar chart that will display top 10 bi grams'''
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_clean_bigrams = (pd.Series(nltk.ngrams(cs_lem, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    top_10_csharp_clean_bigrams.sort_values(ascending=True).plot.barh(color='blue', width=.9, figsize=(10, 6))
    plt.title('10 Most frequently occuring c# bigrams (Cleaned)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def python_bigram_lem(py_lem):
    '''this function will create bar chart that will display top 10 bi grams'''
    # creates bi grams 
    top_10_python_lem_bigrams = (pd.Series(nltk.ngrams(py_lem, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    top_10_python_lem_bigrams.sort_values(ascending=True).plot.barh(color='brown', width=.9, figsize=(10, 6))
    plt.title('10 Most frequently occuring c# bigrams (Lemmatized)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_lem_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def python_bigrams_clean(py_clean):
    '''this function will create bar chart that will display top 10 bi grams'''

    # creates bi grams 
    top_10_python_clean_bigrams = (pd.Series(nltk.ngrams(py_clean, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    top_10_python_clean_bigrams.sort_values(ascending=True).plot.barh(color='black', width=.9, figsize=(10, 6))
    plt.title('10 Most frequently occuring c# bigrams (Cleaned)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def anova_test(js_lem_length, jv_lem_length,cs_lem_length,py_lem_length):
    # statistical test results, 
    return stats.kruskal(js_lem_length, jv_lem_length,cs_lem_length,py_lem_length)


########ALLANTE'S EXPORE CODE######################
