import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings('ignore')

import acquire

############################################
# This is helper file that prepares text for the NLP
############################################

def clean_html_markdown(string: str) -> str:
    '''
    1st cleaning step.
    Removes all links that are html or markdown
    Removes all markdown code
    
    Parameters:
        string to clean
    Returns:
        clean string
    '''
    # create Beautiful Soup object
    s = BeautifulSoup(string)
    # remove all html tags
    #for data in s(['style', 'script']):
    #    data.decompose()
    # save the result to the string
    #string = ' '.join(s.stripped_strings)
    #string = ' '.join(s.findAll(text=True))
    string = s.text
    # replace the code part in markdown, everything that is between ``` and ```
    string = re.sub(r'```.*```', ' ', string, flags=re.DOTALL)
    # remove markdown links
    string = re.sub(r"\]\(.*\)", " ", string, flags=re.DOTALL)
    # remove http links if any left
    string = re.sub(r'http([a-zA-Z0-9\/\:\.\_\-\?\=\&])*\w'," ", string, flags=re.DOTALL)
    string = string.replace('\n', ' ')
    return string

def basic_clean(s:str) -> str:
    '''
    Makes a first basic clean:
        Lowercase everything
        Normalize unicode characters
        Replace anything that is not a letter, number, whitespace or a single quote.
    Parameters:
        s -> string to clean
    Returns:
        s -> cleaned string
    '''
    # all leters to lower case
    s = s.lower()
    # leave only ascii symbols
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    # using regex remove everything that is not a letter a-z, number 0-9, whitespace \s or single quote\'
    s = re.sub('[^a-z0-9\'\s]', '', s)
    
    return s

def tokenize(s:str, return_str: bool = True) -> str or list:
    '''
    Tokenizes all words in the string
    
    Parameters:
        s -> string to be tokenized
        return_list -> boolean:
            if False -> returns list of words
            if True -> returs a tokenized string
    Returns:
        a tokenized string or list of tokenized words
    '''
    # define the Tokenizer
    tokenize = nltk.tokenize.ToktokTokenizer()
    if return_str:
        # returns a string
        return tokenize.tokenize(s, return_str=True)
    else:
        # returns a list of words
        return tokenize.tokenize(s, return_str=False)

def stem(s:str) -> str:
    '''
    Applies stemming to all the words
    
    Parameters:
        s: original string
    Returns:
        s: string with word's stems
    '''
    # define the PorterStemmer
    ps = nltk.porter.PorterStemmer()
    # create a list with stems of words
    stems = [ps.stem(word) for word in s.split()]
    # join the words together as a string where words are separated by whitespace and return it
    return ' '.join(stems)

def lemmatize(s:str) -> str:
    '''
    Applies the lemmatization to each word in the passed string
    
    Parameters:
        s: string
    Returns: string with lemmatized words
    ----
    If the function doesn't work after importing nltk package 
    run nltk.download('all') in order to download all helper files
    '''
    # create a lemmatizer
    wnl = nltk.WordNetLemmatizer()
    # save lemmatized words into a list of words
    lemmas = [wnl.lemmatize(word) for word in s.split()]
    # join the words together as a string where words are separated by whitespace and return it
    return ' '.join(lemmas)

def remove_stopwords(s:str,extra_words:list or str = '', exclude_words:list or str = '') -> str:
    '''
    Obtains the list of stopwords in English. Optional: adds or removes certain words from the list.
    Removes the stopwords from the string.
    
    Parameters:
        s: string, original text were the stopwords should be removed
        extra_words: string, single word or list of strings with words to be added to the stoplist
        exclude_words: string, single word or list of strings with words to be removed from the stopwords list
    Returns:
        s: string, the text with stopwords removed from it
    '''
    # string to lower case
    s = s.lower()
    # create a list of stopwords in English
    stopwords_english = stopwords.words('english')
    
    # extra_words
    # if extra_words is a string, append the word
    if type(extra_words) == str:
        stopwords_english.append(extra_words)
    else: # if it is a list of words
        # add that list of words to list of stopwords
        stopwords_english += extra_words
    
    # exclude_words
    # if exclude_words is a single word string and this words is in stopwords list
    if type(exclude_words) == str and (exclude_words in stopwords_english):
        # remove that word from the stopwords list
        try:
            stopwords_english.remove(exclude_words)
        except ValueError:
            pass  
    # if the exclude_words is a list of words
    if type(exclude_words) == list:
        # for every word remove it from the list
        for word in exclude_words:
            try:
                stopwords_english.remove(word)
            except ValueError:
                pass
    # return a string without stopwords
    return ' '.join([word for word in s.split() if word not in stopwords_english])

####### APPLY FUNCTIONS

def get_clean_df() -> pd.DataFrame:
    '''
    Acquires the data from acquire helper file, saves it into a data frame.
    Cleans columns by appying cleaning functions from this file.
    Return:
        df: pd.DataFrame -> cleaned data frame
    '''

    # acquire a data from inshorts.com website
    df = pd.DataFrame(acquire.scrape_github_data())
    # news_df transformations
    # rename columns
    df.rename({'readme_contents':'original'}, axis=1, inplace=True)
    # create a column 'first_clean' hlml and markdown removed
    df['first_clean'] = df.original.apply(clean_html_markdown)
    # create a column 'clean' lower case, ascii, no stopwords
    df['clean'] = df.first_clean.apply(basic_clean).apply(tokenize).apply(remove_stopwords,extra_words=["'", 'space'])
    # only stems
    #df['stemmed'] = news_df.clean.apply(stem)
    # only lemmas
    df['lemmatized'] = df.clean.apply(lemmatize)

    return df


