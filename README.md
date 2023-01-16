# <b><i><font size="20">What Language Is That?!</font></i></b>
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
|Justin Evans        |[![LinkedIn](https://img.shields.io/badge/Justin's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/qmcbt)|[![GitHub](https://img.shields.io/badge/Justin's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/QMCBT-JustinEvans)|
|Nadia Paz           |[![LinkedIn](https://img.shields.io/badge/Nadia's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadiapaz)|[![GitHub](https://img.shields.io/badge/Nadia's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nadia-paz)|
|Allante Staten      |[![LinkedIn](https://img.shields.io/badge/Allante's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/allantestaten)|[![GitHub](https://img.shields.io/badge/Allante's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/allantestaten)|
|Zachary Stufflebeme |[![LinkedIn](https://img.shields.io/badge/Zachary's%20linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zachary-stufflebeme-63379a243)|[![GitHub](https://img.shields.io/badge/Zachary's%20GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Zachary-Stufflebeme)|

# Project Overview:
This team was tasked to build a model that can predict the main programming language of space themed repositories that are either Python, Java, Javascript, or C# using data from the GitHub repository README files.

We will attempt to accomplish this Using **Natural Language Processing** and following the full *Data Science Pipeline*
We will explore the data and use insights gained to feature engineer our dataset to try and improve our models accuracy
We will create and run  multiple models on the final manipulations of our data on train and validation sets and find our best/most efficient model.
We will than test that model on more outside data and create a function that will take in a readme file text and output the models assignment of the texts main coding language.


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
