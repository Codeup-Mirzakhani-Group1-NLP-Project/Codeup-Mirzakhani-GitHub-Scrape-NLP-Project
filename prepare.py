# analyzing libraries
import pandas as pd
import numpy as np
import unicodedata
# text libraries
import re
import nltk
from bs4 import BeautifulSoup
import nltk.sentiment
from nltk.corpus import stopwords
# modeling preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# helper module
import acquire

# ignore warnings from BeautifulSoup
import warnings
warnings.filterwarnings('ignore')


############################################
# This is a helper file that prepares text for the NLP
############################################

###### Global variables ####################
seed = 42
target = 'language'
############################################

########### FUNCTIONS ######################

####### TEXT PREPROCESSING

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
    s = re.sub('[^a-z\'\s]', '', s)
    
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

def get_clean_df(predictions:bool=False, text:str='') -> pd.DataFrame:
    '''
    Acquires the data from acquire helper file, saves it into a data frame.
    Cleans columns by appying cleaning functions from this file.
    Return:
        df: pd.DataFrame -> cleaned data frame
    '''

    if predictions:
        df = pd.DataFrame(columns=['repo', 'language', 'readme_contents'])
        df.loc[len(df)] = [None, None, text]
    else:
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
    # ENGINEER FEATURES BASED ON THE CLEAN TEXT COLUMN
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    # adds counpound sentiment score
    df['sentiment'] = df['clean'].apply(lambda doc: sia.polarity_scores(doc)['compound'])
    # numerical
    df['lem_length'] = df.lemmatized.str.len()
    df['original_length'] = df.original.str.len()
    df['clean_length'] = df.clean.str.len()
    df['length_diff'] = df.original_length - df.clean_length
    # categorical
    df['has_#9'] = np.where(df.clean.str.contains('&#9;'), 1, 0)
    df['has_parts'] = np.where((df.clean.str.contains(' part ')) | (df.clean.str.contains('parts')), 1, 0)
    df['has_fix'] = np.where(df.clean.str.contains(' fix '), 1, 0)
    df['has_tab'] = np.where(df.clean.str.contains(' tab '), 1, 0)
    df['has_x'] = np.where(df.clean.str.contains(' x '), 1, 0)
    df['has_v'] = np.where(df.clean.str.contains(' v '), 1, 0)
    df['has_codeblock'] = np.where(df.clean.str.contains('codeblock'), 1, 0)
    df['has_image'] = np.where(df.clean.str.contains('image'), 1, 0)
    # change language to category
    df.language = pd.Categorical(df.language)
    # drop repo column
    df.drop('repo', axis=1, inplace=True)
    # drop 'clean_length' columns, as it is part of length_diff column
    df.drop('clean_length', axis=1, inplace=True)
    # reorder columns
    new_order = ['original', 'first_clean', 'clean', 'lemmatized', 'sentiment', 'lem_length',
        'original_length', 'length_diff', 'has_#9', 'has_tab',\
        'has_parts', 'has_fix', 'has_x', 'has_v',\
       'has_codeblock', 'has_image', 'language']
    df = df[new_order]
    return df

####### PREPARATIONS FOR THE MODELING

def scale_numeric_data(X_train, X_validate, X_test):
    '''
    Scales numerical columns.
    Parameters:
        train, validate, test data sets
    Returns:
    train, validate, test data sets with scaled data
    '''
    # features to scale
    to_scale = ['sentiment', 'lem_length', 'original_length',  'length_diff']
    # create a scaler
    sc = MinMaxScaler()
    sc.fit(X_train[to_scale])
    
    # transform data
    X_train[to_scale] = sc.transform(X_train[to_scale])
    X_validate[to_scale] = sc.transform(X_validate[to_scale])
    X_test[to_scale] = sc.transform(X_test[to_scale])
    
    return X_train, X_validate, X_test

####### SPLITTING FUNCTIONS
def split_3(df, explore=True):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    if explore:
        explore_columns = ['original', 'first_clean', 'clean', 'lemmatized', 'sentiment', 'lem_length',\
            'original_length', 'length_diff', 'language']
        df = df[explore_columns]
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed, stratify=train_validate[target])
    return train, validate, test

def split_data(df, explore=True):
    '''
    the function accepts a dataframe as a parameter
    splits according to the purpose
    for the exploration returns train, validate, test
    for modeling it drops unneeded columns, creates dummis, and returns
    6 values X_train, y_train ...
    '''

    if explore:
        return split_3(df)
    else:
        train, validate, test = split_3(df, explore=False)
        train, validate, test = scale_numeric_data(train, validate, test)
        return train.iloc[:, 3:-1], validate.iloc[:, 3:-1], test.iloc[:, 3:-1], \
            train[target], validate[target], test[target]

############ PREPARE DATA FOR MODELING ############

df = get_clean_df()
train, _ , _, _, _, _ = split_data(df, explore=False)
train_ser = train.lemmatized

def get_additional_stopwords(ser: pd.Series = train_ser) -> list:
    '''
    Vectorizes the Series, calculates IDF, creates a list of values where idf score is bigger than 5.65.
    This list can be used as stopwords for creating Bag of Words
    Parameters:
        ser: pandas series or data frame column that contains text
    Returns:
        list of strings -> stopwords
    '''
    tv = TfidfVectorizer()
    tv.fit(ser)
    idf_values = pd.Series(
        dict(
            zip(
                tv.get_feature_names_out(), tv.idf_)))
    # get the list of stop words
    # 5.65 -> sweet spot
    return idf_values[idf_values > 5.65].index.tolist()

def vectorize(train_ser: pd.Series, validate_ser: pd.Series, test_ser: pd.Series, stopwords: list[str]):
    '''
    Applies TfidfVectorizer to text column from train, validate and test data sets.
    Creates Bag of Words
    
    Parameters:
        train_ser: train[column to vectorize]
        validate_ser: validate[column to vectorize]
        test_ser: test[column to vectorize]
        stopwords: list of stopwords that should not be included in the bag of words
    Returns:
        3 data frames train/validate/test with bag of words 
    '''
    # create a vectorizer with stop words
    tv = TfidfVectorizer(stop_words=stopwords)
    # fit transform train 
    train_tv = tv.fit_transform(train_ser)
    # transform validate
    validate_tv = tv.transform(validate_ser)
    # transform test
    test_tv = tv.transform(test_ser)
    
    # create Bag of Words data frames
    # for column names extract features
    # for index us series indexes
    XF_train = pd.DataFrame(train_tv.todense(),
                            columns=tv.get_feature_names_out(),
                           index = train_ser.index)
    XF_validate = pd.DataFrame(validate_tv.todense(), 
                               columns=tv.get_feature_names_out(),
                              index=validate_ser.index)
    XF_test = pd.DataFrame(test_tv.todense(), 
                           columns=tv.get_feature_names_out(),
                          index=test_ser.index)
    
    
    return XF_train, XF_validate, XF_test

def vectorize_for_predictions(stopwords: list[str], text='', train_ser : pd.Series = train_ser):
    '''
    Applies TfidfVectorizer to text column from train, validate and test data sets.
    Creates Bag of Words
    
    Parameters:
        train_ser: train[column to vectorize]
        validate_ser: validate[column to vectorize]
        test_ser: test[column to vectorize]
        stopwords: list of stopwords that should not be included in the bag of words
    Returns:
        3 data frames train/validate/test with bag of words 
    '''
    df = get_clean_df(predictions=True, text=text)

    # create a vectorizer with stop words
    tv = TfidfVectorizer(stop_words=stopwords)
    # fit transform train 
    train_tv = tv.fit_transform(train_ser)
    # transform lemmatized text
    predicions_tv = tv.transform(df.lemmatized)
    
    # create Bag of Words data frames
    # for column names extract features
    # for index us series indexes

    ###
    XF = pd.DataFrame(predicions_tv.todense(), 
                           columns=tv.get_feature_names_out())
                          #index=test_ser.index)
    df = df.iloc[:, 4:-1]
    to_scale = ['sentiment', 'lem_length', 'original_length',  'length_diff']
    # create a scaler
    sc = MinMaxScaler()
    sc.fit(train[to_scale])
    df[to_scale] = sc.transform(df[to_scale])
    
    return pd.concat([XF, df], axis=1)

def get_modeling_data(predictions:bool=False, text:str=''):
    '''
    Calls functions to:
    - get the data frame with the clean text
    - split data
    - get additional stopwords
    - vectorize data
    Concatentes vectorized (bag of words) and numerical columns.
    Returns:
    X_train, X_validate< X_test: data sets for modeling
    y_train, y_validate, y_tes: target variables
    '''
    if predictions:
        df = get_clean_df(predictions=True, text=text)
    else:
        df = get_clean_df()
        # get splitted data sets and target variables
        X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(df, explore=False)
        # create series from lemmatized text column
        train_ser = X_train.lemmatized
        validate_ser = X_validate.lemmatized
        test_ser = X_test.lemmatized
        # separate numerical columns
        train_num = X_train.drop('lemmatized', axis = 1)
        validate_num = X_validate.drop('lemmatized', axis = 1)
        test_num = X_test.drop('lemmatized', axis = 1)
        # create bag of words using vectorize function
        XF_train, XF_validate, XF_test = vectorize(train_ser, 
                                                validate_ser, 
                                                test_ser, 
                                                get_additional_stopwords(train_ser))
        # concatenate bag of words and numerical values 
        X_train_complete = pd.concat([XF_train, train_num], axis=1)                            
        X_validate_complete = pd.concat([XF_validate, validate_num], axis=1)                             
        X_test_complete = pd.concat([XF_test, test_num], axis=1)
        
        return X_train_complete, X_validate_complete, X_test_complete, y_train, y_validate, y_test
