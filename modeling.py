# Data Analysis library
import pandas as pd
import matplotlib.pyplot as plt
# Machine Learning libraries
# model selection
from sklearn.model_selection import GridSearchCV
# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# helper preprocessing module
import prepare as pr

########### GLOBAL VARIABLES #########
#readme to be used for modeling fuction.
test_string = '''# <b><i><font size="20">What Language Is That?!</font></i></b>
#### [Click this link to visit our Project Github](https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project)
#### [Click this link to see our Google Slide Presentation](https://www.canva.com/design/DAFXYtmJecw/zcgBWNIuZw722jgt14ferg/edit)

<!--
## Meet Group 1
|Team Member         |[LinkedIn]                                                |[GitHub]                               |
|:-------------------|:---------------------------------------------------------|:--------------------------------------|
|Justin Evans        |https://www.linkedin.com/in/qmcbt                         |https://github.com/QMCBT-JustinEvans   |
|Nadia Paz           |https://www.linkedin.com/in/nadiapaz                      |https://github.com/nadia-paz           |
|Allante Staten      |https://www.linkedin.com/in/allantestaten                 |https://github.com/allantestaten       |
|Zachary Stufflebeme |https://www.linkedin.com/in/zachary-stufflebeme-63379a243 |https://github.com/Zachary-Stufflebeme |
-->
## Meet Group 1
|Team Member         |[LinkedIn]                                               |[GitHub]                              |
|:-------------------|:--------------------------------------------------------|:-------------------------------------|
|Justin Evans        |[![LinkedIn](https://img.shields.io/badge/Justin's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/qmcbt)|[![GitHub](https://img.shields.io/badge/Justin's%20GitHub-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white)](https://github.com/QMCBT-JustinEvans)|
|Nadia Paz           |[![LinkedIn](https://img.shields.io/badge/Nadia's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadiapaz)|[![GitHub](https://img.shields.io/badge/Nadia's%20GitHub-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white)](https://github.com/nadia-paz)|
|Allante Staten      |[![LinkedIn](https://img.shields.io/badge/Allante's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/allantestaten)|[![GitHub](https://img.shields.io/badge/Allante's%20GitHub-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white)](https://github.com/allantestaten)|
|Zachary Stufflebeme |[![LinkedIn](https://img.shields.io/badge/Zachary's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zachary-stufflebeme-63379a243)|[![GitHub](https://img.shields.io/badge/Zachary's%20GitHub-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white)](https://github.com/Zachary-Stufflebeme)|

# Project Overview:
This team was tasked to build a model that can predict the main programming language of a repository using data from the GitHub repository README files.

Using **Natural Language Processing** and following the full *Data Science Pipeline*


# Project Goals:
* Produce a Final GitHub repository containing our work
* Provide a well-documented jupyter notebook that contains our analysis
* Display a `README` file that contains a description of our project and instructions on how to run it with a link to our Google Slide Presentation
* Present a google slide deck suitable for a general audience which summarizes our findings in exploration and documents the results of our modeling
with well-labeled visualizations
* Produce and demonstrate a function that will take in the text of a `README` file, and attempt to predict the programming language using our best model.

# Reproduction of this Data:
* Can be accomplished using a local `env.py` containing `github_username`, `github_token`, and host Repository link information for access to the GitHub project Readme file search results that you want to explore.
**Warning** to make the scraping successfull we added pauses 20 sec/per page. This slows down the first run of the program. After the scraping all data is saved locally in the `data.json` file.
* To retrieve a github personal access token:
* 1. Go here and generate a personal access token: https://github.com/settings/tokens
You do _not_ need to select any scopes, i.e. leave all the checkboxes unchecked
* 2. Save it in your env.py file under the variable ```github_token```
Add your github username to your env.py file under the variable ```github_username```

* Clone the Repository using this code ```git clone git@github.com:Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project.git``` then run the ```Final_Report_NLP-Project.ipynb``` Jupyter Notebook. You will need to ensure the below listed files, at a minimum, are included in the repo in order to be able to run.
* `Final_Report_NLP-Project.ipynb`
* `acquire.py`
* `prepare.py`
* `explore_final.py`
* `modeling.py`

* A step by step walk through of each piece of the Data Science pipeline can be found by reading and running the support files located in the individual team members folders on our ```Codeup-Mirzakhani-GitHub-Scrape-NLP-Project``` github repository found here: https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project

# Initial Thoughts
Our initial thoughts were that since we centered our `GitHub` repositories around the topic of **Space**, that possibly unique scientific terms found within the readme files would be deterministic of the primary coding language used to conduct exploration and modeling of those projects. Another thought was that the readme files would be peppered with code specific terminology that would reveal the primary language used to code the projects.

# The Plan
* Acquire data from `GitHub` `Readme` files by scraping the `Github API`
* Clean and Prepare the data using `RegEx` and `Beautiful soup`.
* Explore data in search of relevant keyword grouping using bi-grams and n-grams
* Answer the following initial questions:

* **Question 1.** How is the target variable represented in the sample?

* **Question 2.** Are there any specific words or word groups that can assist with identifying the Language JavaScript or Java over the other languages?

* **Question 3.** What are the top words used in cleaned C#?

* **Question 4.** What are the most used words in cleaned python strings?

* **Question 5.** Is there an association between coding language and the lemmatized mean length of the string?

* **Question 6.** Is there a significant difference in Sentiment across all four languages?

* **Question 7.** How different are the bi-grams among four programming languages?


* Develop a Model to predict program language of space related projects using either `Python`, `Javascript`, `Java`, or `C#` based on input from `GitHub` Project `Readme` files.
* Evaluate models on train and validate data using accuracy score
* Select the best model based on the smallest difference in the accuracy score on the train and validate sets.
* Evaluate the best model on test data
* Run Custom Function on a single random Data input from `GitHub` `Readme` file to predict program language of that project.
* Draw conclusions

# Data Dictionary:


## Features
|Feature    |Description       |
|:----------|:-----------------|
|`original`| The original data we pulled from the `GitHub`|
|`first_clean`| Text after cleaning the `html` and `markdown` code|
|`clean`|Tokenized text in lower case, with latin symbols only|
|`lemmatized`|Lemmatized text|
|`sentiment`|The coumpound sentiment score of each observation|
|`lem_length`|The length of the lemmatized text in symbols|
|`original_length`|The length of the original text in symbols|
|`length_diff`|The difference in length between the orignal_length and the length of the `clean` text|
||**Target variable:**|
|`language`|`JavaScript`, `C#`, `Java` or `Python` programming languages|


# Acquire

* We scraped our data from `github.com` using `Beautiful Soup`.
* We grabbed the link of **space themed repos** where the main coding language was either `Python`, `C#`, `Java` or `Javasript` on the first 100 pages of `github`.
* Each row represents a `Readme` file from a different project repository.
* Each column represents a feature created to try and predict the primary coding languge used.
We acquired 432 entries.

# Prepare

**Prepare Actions:**

* **NULLS:** There were no null values all repositories contained a readme for us to reference
* **FEATURE ENGINEER:** Use exploration with bag of words to create new  categorical features from polarizing words. We created columns with `clean` text ,`lemmatized` text , and columns containing the lengths of them as well. We also created a column that we filled with the sentiment score of the text in the readme.
* **DROP:** All Data acquired was used.
* **RENAME:** Columns for Human readability.
    * **REORDER:** Rearange order of columns for convenient manipulation.
    * **DROP 2:** Drop Location Reference Columns unsuitable for use with ML without categorical translation.
* **ENCODED:** No encoding required.
* **MELT:** No melts needed.


# Summary of Data Cleansing
* Luckily all of our data was usable so we had 0 nulls or drops.

* Note: Special care was taken to ensure that there was no leakage of this data. All code parts were removed


# Split

* **SPLIT:** train, validate and test (approx. 50/30/20), stratifying on target of `language`
* **SCALED:** We scaled all numeric columns. ['lem_length','original_length','clean_length','length_diff']
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (features) and y (target)


## A Summary of the data

### There are 432 records (rows) in our training data consisting of 1621 features (columns).
* There are 1618 categorical features
* There are 4 continuous features that represent measurements of value, size, time, or ratio.


# Explore

* In the exploration part we tried to identify if there are words, bigrams or trigrams that could help our model to identify the programming language.
* We ran statistical tests on the numerical features that we have created.
* Explore differences between cleaned and lemmatized versions of c# and python.
* Explore association between coding language and the lemmatized mean of string lengths.

## Exploration Summary of Findings:
* In the space thematic Javascript is the most popular language. It makes up 35% of the data sample.
* Most popular "word" in **C#** is `&#9;`.
* The word `codeblock` appears only in **Python** repositories.
* Most used in **Python** is `python`.
* The words that identifies **Java** most are `x` and `planet`.
* Most appearing bigram in **Javascript** is "bug fixed".
* Bi-grams different a lot among the programming languages `Readme` files, but the number of most occuring bi-grams is not big enough to use them in our modeling.
* There is *no significant difference* in the length of the lemmatized text among the languages.
* There is *no significant difference* in the compound sentiment score among the languages.

# Modeling

### Features that will be selected for Modeling:
* All continious variables:
- `sentiment`
- `lem_length`
- `original_length`
- `length_diff`
* `lemmatized` text turned into the Bag of Words with `TDIFVectorizer`

### Features we didn't include to modeling
* `original`
* `first_clean`
* `clean`

Those features were used in the exploration and do not serve for the modeling.
    N-grams were not created for the modeling.

**The models we created**

We used following classifiers (classification algorithms):
- Decision Tree,
- Random Forest,
- Logistic Regression,
- Gaussian NB,
- Multinational NB,
- Gradient Boosting, and
- XGBoost.

For most of our models we have used `GridSearchCV` algorithm that picked the best feature combinations for our training set. The parameters that we've used you can see below.

To evaluate the models we used the accuracy score. The good outcome is the one where the `accuracy score` is higher than our `baseline` - the propotion of the most popular programming language in our train data set. It is `JavaScript` and `0.35`. So our baseline has the accuracy score - 0.35

## Modeling Summary:
- The best algorithm  is `Random Forest Classifier` with following parameters `{'max_depth': 5, 'min_samples_leaf': 3}`
- It predicts the programming language with accuracy:
    - 63% on the train set
    - 48% on the validate set
    - 59% on the test set
- It makes 24% better predictions on the test set that the baseline model.


# Conclusions: 
## **Exploration:**
* In the space thematic Javascript is the most popular language. It makes up 35% of the data sample.
* Most popular "word" in **C#** is `&#9;`.
* The word `codeblock` appears only in **Python** repositories.
* Most used in **Python** is `python`.
* The words that identifies **Java** most are `x` and `planet`.
* Most appearing bigram in **Javascript** is "bug fixed".
* Bi-grams different a lot among the programming languages `Readme` files, but the number of most occuring bi-grams is not big enough to use them in our modeling.
* There is *no significant difference* in the length of the lemmatized text among the languages.
* There is *no significant difference* in the compound sentiment score among the languages.

## **Modeling:**
* We have created the model that showed 59% accuracy on the test set.
* The results of the model performance are not consistant as the texts from the Readme files don't have a standard and programmers not always describe their work process step by step. That possibly could help our model to pick the outcome in better way.
## **Recommendations:**
* WORDS
* WORDS
* WORDS
## **Next Steps:**
* Retrieve more data to train the model on and potentially identify better features for the model
    * WORDS
'''
# random state seed for split and classification algorithms
seed = 42
# DataFrame to store the scores
scores = pd.DataFrame(columns=['model_name', 'train_score', 'validate_score', 'score_difference'])
# create sets and taget variables for modeling
X_train, X_validate, X_test, y_train, y_validate, y_test = pr.get_modeling_data()
# calculate a baseline
baseline = round(y_train.value_counts(normalize=True)[0], 2)

# some models don't accept text as a target
# create a map to digitize target variable
lang_map = {'Java':0, 'C#':1, 'JavaScript':2, 'Python':3}
y_train_numeric = y_train.map(lang_map)
y_validate_numeric = y_validate.map(lang_map)
y_test_numeric = y_test.map(lang_map)

############# FUNCTIONS TO RUN CLASSIFIERS #############

##### Decision Tree ########
def run_decision_tree(cv:int=5):
    '''
    Classifier: Decision Tree algorithm
    Creates a dictionary of parameters for the classifier
        Uses GridSearchCV to find the best combination of parameters
    Prints the selected parameters on the screen
    Fits the classifier with best parameters to the training set using GridSearch
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    -----------
    Parameters:
        cv: integer, number of cross validation folds for the grid search
    No returns
    '''
    # create a dictionary of parameters
    DTC_parameters = {'max_depth':[ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 ]
                     }
    # create a classifier
    DTC = DecisionTreeClassifier(random_state=seed)
    # creat a grid search
    grid_DTC = GridSearchCV(estimator=DTC, param_grid=DTC_parameters, cv=cv, n_jobs=-1)
    # fit on train set
    grid_DTC.fit(X_train, y_train)
    # print the best parameter's comination
    print(f'Best parameters per algorithm:')
    print('----------------------------------------------------')
    print(f'Decision Tree Parameters:  {grid_DTC.best_params_}')
    # calculate scores
    train_score = grid_DTC.best_estimator_.score(X_train, y_train)
    validate_score = grid_DTC.best_estimator_.score(X_validate, y_validate)
    # save the scores into a dataframe
    scores.loc[len(scores)] = ['Decision Tree', train_score, validate_score, train_score - validate_score]

##### Random Forest ########
def run_random_forest(cv:int=5):
    '''
    Classifier: Random Forest algorithm
    Creates a dictionary of parameters for the classifier
    Uses GridSearchCV to find the best combination of parameters
    Prints the selected parameters on the screen
    Fits the classifier with best parameters to the training set using GridSearch
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    -----------
    Parameters:
        cv: integer, number of cross validation folds for the grid search
    No returns
    '''
    # create a dictionary of parameters
    rf_parameters = {'max_depth':[5, 6, 7],
                     'min_samples_leaf':[2, 3, 5]
                     }
    # create a classifier
    rf = RandomForestClassifier(random_state=seed)
    # creat a grid search
    grid_rf = GridSearchCV(estimator=rf, param_grid=rf_parameters, cv=cv, n_jobs=-1)
    # fit on train set
    grid_rf.fit(X_train, y_train)
    # print the best parameter's comination
    print(f'Random Forest Parameters:  {grid_rf.best_params_}')
    # calculate scores
    train_score = grid_rf.best_estimator_.score(X_train, y_train)
    validate_score = grid_rf.best_estimator_.score(X_validate, y_validate)
    # save the scores into a dataframe
    scores.loc[len(scores)] = ['Random Forest', train_score, validate_score, train_score - validate_score]

##### Logistic Regression and Gaussian NB ########
def run_other():
    '''
    Classifier #1: Logistic Regression
    Classifier #2: Gaussian Naive Bayes
    Creates Logistic Regression and Gaussian NB, fits them on the training set
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    -----------
    No parameters
    No returns
    '''
    ##### STEP 1
    # create Logistic Regression
    lr = LogisticRegression(random_state=seed)
    # fit on the train set
    lr.fit(X_train, y_train)
    # calculate scores
    train_score_lr = lr.score(X_train, y_train)
    validate_score_lr = lr.score(X_validate, y_validate)
    # print params
    print('Logistic Regression: default paramters')
    # save the scores
    scores.loc[len(scores)] = \
        ['Logistic Regression', train_score_lr, validate_score_lr, train_score_lr - validate_score_lr]
    
    ##### STEP 2
    # create Gaussian NB
    nb = GaussianNB()
    # fit on the train set
    nb.fit(X_train, y_train)
    # calculate scores
    train_score_nb = nb.score(X_train, y_train)
    validate_score_nb = nb.score(X_validate, y_validate)
    # print params
    print('Gaussian NB parameters: default paramters')
    # save the scores
    scores.loc[len(scores)] = \
        ['Gaussian Naive Bayes', train_score_nb, validate_score_nb, train_score_nb - validate_score_nb]

##### Multinomial NB ########
def run_multinomial_nb(cv:int=3):
    '''
    Classifier: Multinomial Naive Bayes
    Creates a dictionary of parameters for the classifier 
    Uses GridSearchCV to find the best combination of parameters
    Prints the selected parameters on the screen
    Fits the classifier with best parameters to the training set using GridSearch
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    -----------
    Parameters:
        cv: integer, number of cross validation folds for the grid search
    No returns
    '''
    mnb_parameters = {'alpha': [0.2, 0.5, 1.0]}
    mnb = MultinomialNB(alpha=0.2)
    grid_mnb = GridSearchCV(estimator=mnb, param_grid=mnb_parameters, cv=cv, n_jobs=-1)
    # fit on the train set
    grid_mnb.fit(X_train, y_train)
    # print parameters
    print(f'Multinomial NB Parameters:  {grid_mnb.best_params_}')
    # calculate scores
    train_score = grid_mnb.best_estimator_.score(X_train, y_train)
    validate_score = grid_mnb.best_estimator_.score(X_validate, y_validate)
    # add scores to the dataframe
    scores.loc[len(scores)] = ['Multinomail Naive Bayes', train_score, validate_score, train_score - validate_score]

##### Gradient Boosting ########
def run_gradient_boosting(cv=3):
    '''
    Classifier: Gradient Boosting
    Creates a dictionary of parameters for the classifier 
    Uses GridSearchCV to find the best combination of parameters
    Prints the selected parameters on the screen
    Fits the classifier with best parameters to the training set using GridSearch
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    -----------
    Parameters:
        cv: integer, number of cross validation folds for the grid search
    No returns
    '''
    # create a dictionary of parameters
    gb_parameters = {
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 6, 7],
    }
    # create a classifier
    gb = GradientBoostingClassifier(random_state=seed)
    # create a grid search
    grid_gb = GridSearchCV(estimator=gb, param_grid=gb_parameters, cv=cv, n_jobs=-1)
    # fit on train set
    grid_gb.fit(X_train, y_train)
    # print the best parameter's comination
    print(f'Gradient Boosting Parameters:  {grid_gb.best_params_}')
    # calculate scores
    train_score = grid_gb.best_estimator_.score(X_train, y_train)
    validate_score = grid_gb.best_estimator_.score(X_validate, y_validate)
    # save scores 
    scores.loc[len(scores)] = ['Gradient Boosting', train_score, validate_score, train_score - validate_score]

def run_xgboost(cv=3):
    '''
    Classifier: XGBoost
    Creates a dictionary of parameters for the classifier 
    Uses GridSearchCV to find the best combination of parameters
    Prints the selected parameters on the screen
    Fits the classifier with best parameters to the training set using GridSearch
    Calculates accuracy scores for the training and validate sets and saves them into scores dataframe
    Prints Feature Importance bar chart
    -----------
    Parameters:
        cv: integer, number of cross validation folds for the grid search
    No returns
    '''
    # create parameters
    xb_parameters = {
        'max_depth': [3, 4, 5, 6],
        'gamma': [0.1, 0.2, 0.3]
    }
    # create a classifier
    xb = xgb.XGBClassifier(n_estimators=100,eval_metric='merror',seed=seed)
    # create a grid search
    grid_xb = GridSearchCV(estimator=xb, param_grid=xb_parameters, cv=cv, n_jobs=-1)
    # fit on the train set
    grid_xb.fit(X_train, y_train_numeric)
    # print parameters
    print(f'XGBoost Parameters:  {grid_xb.best_params_}')
    print()
    # calculate scores
    train_score = grid_xb.best_estimator_.score(X_train, y_train_numeric)
    validate_score = grid_xb.best_estimator_.score(X_validate, y_validate_numeric)
    # add scores to the dataframe
    scores.loc[len(scores)] = ['XGBoost', train_score, validate_score, train_score - validate_score]


############# RUN ALL CLASSIFIERS #############
def run_all_classifiers() -> pd.DataFrame:
    '''
    Runs all classifiers.
    No return values
    '''
    run_decision_tree()
    run_random_forest()
    run_other() # 2 classifiers
    run_multinomial_nb()
    run_gradient_boosting()
    run_xgboost()

def display_scores():
    '''
    Displays scores
    No return values
    '''
    display(scores.sort_values(by='score_difference'))

def display_feature_importance():
    '''
    Creates an XGBoost Classifier and uses its method 
    plot_importance to show the feature importance
    '''
    # create a braplot and display it
    # we create the classifier again using the best parameters {'gamma': 0.2, 'max_depth': 4}
    xb = xgb.XGBClassifier(gamma=0.2, max_depth=4 ,n_estimators=100,eval_metric='merror',seed=seed)
    xb.fit(X_train, y_train_numeric)
    xgb.plot_importance(xb, max_num_features=7)
    plt.rcParams['figure.figsize'] = [10,7]
    plt.show()

def run_best_model():
    rf = RandomForestClassifier(max_depth=5, min_samples_leaf=2, random_state=seed)
    rf.fit(X_train, y_train)
    #calculate scores
    train_score = round(rf.score(X_train, y_train), 2)
    validate_score = round(rf.score(X_validate, y_validate), 2)
    test_score = round(rf.score(X_test, y_test), 2)
    yhat=rf.predict(X_test)
    return pd.DataFrame({'result':['Random Forest', train_score, validate_score, test_score]},\
                          index=['Model name', 'Train score', 'Validate score', 'Test score'])
#####################################################

def predict_text(text:str):
    '''
    Creates a data Frame out of the text, prepares this data for predictions, predicts outcome
    Parameters:
        text: string with a text from README file
    Returns:
        str: predictions: JavaScript, Java, C# or Python
    '''
    to_predict = pr.vectorize_for_predictions(pr.get_additional_stopwords(), text=text)
    rf = RandomForestClassifier(max_depth=5, min_samples_leaf=2, random_state=seed)
    rf.fit(X_train, y_train)
    return rf.predict(to_predict)




