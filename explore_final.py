###################
##### IMPORTS #####
###################

import os
import re
import json
import requests
import unicodedata
import pandas as pd
import scipy.stats as stats
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union, cast

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

## Plots, Graphs, & Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# ------------- #
# Local Imports #
# ------------- #

from env import github_username, github_token
import acquire as ac
import prepare as pr



#############################
##### Acquire & Prepare #####
#############################

df = pr.get_clean_df()
train, validate, test = pr.split_data(df, explore=True)
target = 'language'
seed = 42



######################################
##### Allante's Global Variables #####
######################################

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



#####################################
##### Justin's Global Variables #####
#####################################

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



#######################################
##### Allante's Explore Functions #####
#######################################

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

# STAT TEST
def kruskal_test(js_lem_length, jv_lem_length,cs_lem_length,py_lem_length):
    # statistical test results, 
    return stats.kruskal(js_lem_length, jv_lem_length,cs_lem_length,py_lem_length)

######### COMPARE BI-GRAMS #############
def csharp_java_bigrams(cs_clean, jv_clean):
    '''
    creates sublots to display side_by_side
    c# and java bigrams
    '''
    # creates c sharp 10 most frequent bigrams 
    top_10_java_bigrams = (pd.Series(nltk.ngrams(jv_clean, 2)).value_counts().head(10))
    # sorts 
    
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_bigrams = (pd.Series(nltk.ngrams(cs_clean, 2)).value_counts().head(10))
 
    # plot bigrams
    plt.figure(figsize=(24, 6))
    plt.rc('font', size=14)
    plt.suptitle('10 Most frequently occuring bigrams')

    # 1st subplot C#
    plt.subplot(121)
    top_10_csharp_bigrams.plot.barh(color='blue', width=.9)
    plt.title('C#')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.gca().invert_yaxis()

    # 2nd subplot Java bigrams
    plt.subplot(122)
    top_10_java_bigrams.plot.barh(color='green', width=.9)
    plt.title('Java')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_java_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.gca().invert_yaxis()

    plt.show()

def python_js_bigrams(py_clean, js_clean):
    '''
    creates sublots to display side_by_side
    javascript and python bigrams
    '''
    # creates c sharp 10 most frequent bigrams 
    top_10_js_bigrams = (pd.Series(nltk.ngrams(js_clean, 2)).value_counts().head(10))
    # sorts 
    
    # creates c sharp 10 most frequent bigrams 
    top_10_python_bigrams = (pd.Series(nltk.ngrams(py_clean, 2)).value_counts().head(10))
    
    # plot bigrams
    plt.figure(figsize=(24, 6))
    plt.rc('font', size=14)
    plt.suptitle('10 Most frequently occuring bigrams')

    # 1st subplot Javascript
    plt.subplot(121)
    top_10_js_bigrams.plot.barh(color='indianred', width=.9)
    plt.title('Javascript')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_js_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.gca().invert_yaxis()

    # 2nd subplot Python bigrams
    plt.subplot(122)
    top_10_python_bigrams.plot.barh(color='darkviolet', width=.9)
    plt.title('Python')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    plt.gca().invert_yaxis()

    plt.show()
############## END OF COMPARE BI-GRAMS ###########

############## LEMMATIZED VS CLEAN BI-GRAMS ###########

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


def csharp_bigrams_clean(cs_clean):
    '''this function will create bar chart that will display top 10 bi grams'''
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_clean_bigrams = (pd.Series(nltk.ngrams(cs_clean, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    top_10_csharp_clean_bigrams.sort_values(ascending=True).plot.barh(color='blue', width=.9, figsize=(10, 6))
    plt.title('10 Most frequently occuring c# bigrams (Cleaned)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def csharp_bigrams(cs_lem, cs_clean):
    '''
    creates sublots to display side_by_side
    '''
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_lem_bigrams = (pd.Series(nltk.ngrams(cs_lem, 2)).value_counts().head(10))
    # sorts 
    
    # creates c sharp 10 most frequent bigrams 
    top_10_csharp_clean_bigrams = (pd.Series(nltk.ngrams(cs_clean, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
 

    plt.figure(figsize=(24, 6))
    plt.rc('font', size=14)
    plt.suptitle('10 Most frequently occuring c# bigrams')
    plt.subplot(121)
    top_10_csharp_clean_bigrams.sort_values(ascending=True).plot.barh(color='blue', width=.9)
    plt.title('Cleaned')
    #plt.ylabel('Bigram')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

    plt.subplot(122)
    top_10_csharp_lem_bigrams.sort_values(ascending=True).plot.barh(color='green', width=.9)
    plt.title('Lemmatized')
    #plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_csharp_lem_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)
    # set the spacing between subplots
    #fig.tight_layout()
    plt.show()

def python_bigram_lem(py_lem):
    '''this function will create bar chart that will display top 10 bi grams'''
    # creates bi grams 
    top_10_python_lem_bigrams = (pd.Series(nltk.ngrams(py_lem, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    top_10_python_lem_bigrams.sort_values(ascending=True).plot.barh(color='brown', width=.9, figsize=(10, 6))
    plt.title('10 Most frequently occuring python bigrams (Lemmatized)')
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
    plt.title('10 Most frequently occuring python bigrams (Cleaned)')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

def python_bigrams(py_lem, py_clean):
    '''
    displays top10 python bigrams
    '''
    # creates bi grams 
    top_10_python_lem_bigrams = (pd.Series(nltk.ngrams(py_lem, 2)).value_counts().head(10))
    # creates bi grams 
    top_10_python_clean_bigrams = (pd.Series(nltk.ngrams(py_clean, 2)).value_counts().head(10))
    # sorts bi grams and provides bar gram and color of bars
    plt.figure(figsize=(24, 6))
    #plt.rc('font', size=14)
    plt.suptitle('10 Most frequently occuring Python bigrams')
    plt.subplot(121)
    top_10_python_lem_bigrams.sort_values(ascending=True).plot.barh(color='brown', width=.9)
    plt.title('Lemmatized')
    #plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_lem_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)


    # sorts bi grams and provides bar gram and color of bars
    plt.subplot(122)
    top_10_python_clean_bigrams.sort_values(ascending=True).plot.barh(color='black', width=.9)
    plt.title('Cleaned')
    #plt.ylabel('Bigram')
    plt.xlabel('# Occurances')
    # plotting tick marks and resetting index
    ticks, _ = plt.yticks()
    labels = top_10_python_clean_bigrams.reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1])
    _ = plt.yticks(ticks, labels)

############## END OF LEMMATIZED VS CLEAN BI-GRAMS ###########




######################################
##### Justin's Explore Functions #####
######################################

def qmcbt_viz_01():
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
    # sort by one language
    return lem_word_counts.sort_values(['JavaScript'], ascending=False)[1:2]
    
def qmcbt_viz_02():
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
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
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
    # sort by one language
    return lem_word_counts.sort_values(['Java'], ascending=False).head(1)

def qmcbt_viz_04():
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
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
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
    # Display top Bi-Gram pair for JavaScript lemmatized
    pd.Series(js_2_gram).value_counts().head(5).plot.barh()
    plt.title('JavaScript Bi-Grams')
    plt.gca().invert_yaxis()
    return plt.show()

def qmcbt_viz_06():
    """
    This Function Displays a Visualization needed for the Final Presentation.
    """
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
    """
    Description:
    This Function Displays the results of an ANOVA Statistical Test.
    
    Required Imports:
    import scipy.stats as stats
    
    Arguments:
    NONE
    
    """
        
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
