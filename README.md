# <b><i><font size="20">What Language Is That?!</font></i></b>
#### [Click this link to visit our Project Github](https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project)
#### [Click this link to see our Google Slide Presentation](https://www.canva.com/design/DAFXYtmJecw/zcgBWNIuZw722jgt14ferg/edit)

## Meet Group 1
|Team Member         |[LinkedIn]                                                |[GitHub]                               |
|:-------------------|:---------------------------------------------------------|:--------------------------------------|
|Justin Evans        |https://www.linkedin.com/in/qmcbt                         |https://github.com/QMCBT-JustinEvans   |
|Nadia Paz           |https://www.linkedin.com/in/nadiapaz                      |https://github.com/nadia-paz           |
|Allante Staten      |https://www.linkedin.com/in/allantestaten                 |https://github.com/allantestaten       |
|Zachary Stufflebeme |https://www.linkedin.com/in/zachary-stufflebeme-63379a243 |https://github.com/Zachary-Stufflebeme |

# Project Overview:
This team was tasked to build a model that can predict the main programming language of a repository using data from the GitHub repository README files.

Using Natural Language Processing and following the full Data Science Pipeline 


# Project Goals:
* Produce a Final GitHub repository containing our work
* Provide a well-documented jupyter notebook that contains our analysis
* Display a README file that contains a description of our project and instructions on how to run it with a link to our Google Slide Presentation
* Present a google slide deck suitable for a general audience which summarizes our findings in exploration and documents the results of your modeling
with well-labeled visualizations
* Produce and demonstrate a Function that will take in the text of a README file, and attempt to predict the programming language using our best model.

# Reproduction of this Data:
* Can be accomplished using a local env.py containing github_username, github_token, and host Repository link information for access to the GitHub project Readme file search results that you want to explore.
   * TODO: Make a github personal access token.
      * 1. Go here and generate a personal access token: https://github.com/settings/tokens
        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
      * 2. Save it in your env.py file under the variable `github_token`
        TODO: Add your github username to your env.py file under the variable `github_username`
        TODO: Add more repositories to the `REPOS` list below.
* All other step by step instructions can be found by reading and running the below Jupyter Notebook file located in our Codeup-Mirzakhani-GitHub-Scrape-NLP-Project github repository found here:
   * https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project
   * Final_Report_NLP-Project.ipynb
   * acquire.py
   * prepare.py
   * explore.py
   * model.py
    
# Initial Thoughts
Our initial thoughts were that since we centered our GitHub repositories around the topic of Space, that possibly unique scientific terms found within the readme files would be deterministic of the primary coding language used to conduct exploration and modeling of those projects. Another thought was that the readme files would be peppered with code specific terminology that would reveal the primary language used to code the projects.

# The Plan
* Acquire data from GitHub Readme files by scraping the Github API
* Clean and Prepare the data using RegEx
* Explore data in search of relevant keyword grouping using bi-grams and n-grams 
* Answer the following initial question:

    * **Question 1.** INSERT_QUESTION_HERE? 

    * **Question 2.** INSERT_QUESTION_HERE?

    * **Question 3.** INSERT_QUESTION_HERE? 

    * **Question 4.** INSERT_QUESTION_HERE?

    * **Question 5.** INSERT_QUESTION_HERE?

* Develop a Model to predict program language based on inut from GitHub Project Readme files.
    * Use drivers identified in explore to build predictive models of error using...
    * Evaluate models on train and validate data using RMSE (Root mean square Error)
    * Select the best model based on the least RMSE
    * Evaluate the best model on test data
* Run Custom Function on a single random Data input from GitHub Readme file to predict program language of that project.
* Draw conclusions


# Data Dictionary:

    
## Features
|Feature    |Description       |
|:----------|:-----------------|
|original||	
|first_clean||
|clean||
|lemmatized||
|sentiment||
|lem_length||
|original_length||
|length_diff||
|has_#9||
|has_tab||
|has_parts||
|has_fix||
|has_x||
|has_v||
|has_codeblock||
|has_image||
|language||


# Acquire

* ```zillow``` data from Codeup SQL database was used for this project.
* The data was initially pulled on 15-NOV-2022.
* The initial DataFrame contained 52,441 records with 69 features  
    (69 columns and 52,441 rows) before cleaning & preparation.
* Each row represents a Single Family Property record with a Tax Asessment date within 2017.
* Each column represents a feature provided by Zillow or an informational element about the Property.


# Prepare

**Prepare Actions:**

* **Whitespace:** Removed 52,441 Whitespace characters.
* **REFORMAT:** Reformatted 13 columns containing 596,382 NaN entries to 0.
* **CONVERT dtypes:** Convert dtypes to accurately reflect data contained within Feature.
* **FEATURE ENGINEER:** Use Yearbuilt to create Age Feature, Drop yearbuilt for redundancy; create Feature to show ratio of Bathrooms to Bedrooms.
* **fips CONVERSION:** Use fips master list to convert fips to county and state, Drop state for redundancy.
* **PIVOT:** Pivot the resulting county column from fips conversion to 3 catagorical features. 
* **DROP:** Dropped 27 Columns unecessary to data prediction (ie.. index and redundant features).
* **REPLACE:** Replaced conditional values in 2 columns to transform into categorical features.
* **RENAME:** Columns for Human readability.    
* **REORDER:** Rearange order of columns for human readability.   
* **DROP 2:** Drop Location Reference Columns unsuitable for use with ML without categorical translation.
* **CACHE:** Write cleaned DataFrame into a new csv file ('zillow_2017_cleaned.csv').  
* **ENCODED:** No encoding required.
* **MELT:** No melts needed.


# Summary of Data Cleansing
* Cleaning the data resulted in less than 10% overall record loss

* DROP NaN COLUMNS: 39 features each containing over 30% NaN were dropped; resulting in no record loss.
*    DROP NaN ROWS: 1,768 records containing NaN across 13 features were dropped; resulting in only 3% record loss.
*         OUTLIERS: Aproximately 3,000 outliers were filtered out in an attempt to more accurately align with realistic
                    expectations of a Single Family Residence; resulting in less than a 6% decrease in overall records.
*           IMPUTE: No data was imputed

* logerror: The original logerreror prediction data was pulled over and prepared with this DataFrame for later comparison in order to meet the requirement of improving the original model.  
    
* Note: Special care was taken to ensure that there was no leakage of this data.


# Split

* **SPLIT:** train, validate and test (approx. 50/30/20), stratifying on target of 'language'
* **SCALED:** no scaling was conducted
* **Xy SPLIT:** split each DataFrame (train, validate, test) into X (selected features) and y (target) 


## A Summary of the data

### There are 28,561 records (rows) in our training data consisting of 18 features (columns).
* There are 7 categorical features made up of only 2 unique vales indicating True/False.
* There are 5 categorical features made up of multiple numeric count values.
* There are 6 continuous features that represent measurements of value, size, time, or ratio.


# Explore

* Exploration of the data was conducted using various Correlation Heat Maps, Plot Variable Pairs, Categorical Plots, and many other graph and chart displays to visualize Relationships between independent features and the target as well as their relationships to eachother. 

    
* Each of the three selected features were tested for a relationship with our target of Tax Assesed Value.
    1. Bedrooms
    2. Bathrooms
    3. Property Squarefeet  
    
    
* All three independent features showed a significant relationship with the target feature.

* Three statistical tests were used to test these questions.
    1. T-Test
    2. Pearson's R
    3. $Chi^2$

# Takeaways and Conclusions

## Exploration Summary of Findings:
* Gender seems to have an impact on life expectancy
* Both Male and Female life expectancies raise over the years
* In general year over year the life expectancy rate seems to maintain an upward trend
* Women have a higher life expectany than men

### Features that will be selected for Modeling:
* Our target feature is Tax Assessed Property Value ```('taxvaluedollarcnt')```
* Our selected features are:
    1. Property Squarefeet ```('calculatedfinishedsquarefeet')```
    2. Bathrooms ```('bathroomcny')```
    3. Bedrooms ```('bedroomcnt')```

### Features I'm not moving to modeling with

* Tenure 
* Monthly Charges
* Tech Support

## Modeling Summary:
* All models did slightly better than baseline.
* None of the models were within acceptable proximity to actual target results

* Our top model ```Simple Linear Regression Model```  was run on test data and performed better than baseline as expected and even outperformed its previous score on validation by approximately three base points.

**For this itteration of modeling we have a model that beats baseline.**    

## Comparing Models

* None of the models are anywhere close to being in danger of overfit
* Both of the polynomial models performed at the top
* The Lasso Lars model was not that far behind the polynomials
* Simple LM was a little farther behind but still fairly close
* Baselin and GLM were nearly identical and both performed with a significantly higher rate of error
* logerror did not even beat baseline
    
## ```2nd Degree Polynomial``` is the best model and will likely continue to perform well above Baseline and logerror on the Test data.

# Evaluate on Test: Best Model (```2nd Degree Polynomial```)


# Conclusions: 
* **Exploration:** 
    * We asked 4 Questions using T-Test and Anova Statistical testing to afirm our hypothesis
    * In general year over year the life expectancy rate seems to maintain an upward trend
    * Women have a higher life expectany than men
* **Modeling:**
    * We trained and evaluated 6 different Linear Regression Models, all of which outperformed baseline 
    * We chose the Simple Linear Regression Model as our best performing model
    * When evaluated on Test, it continued to outperform baseline and surpased its previous performance on validate
* **Recommendations:**
    * I think we should hold off on deploying this model.
    * Even though it beat baseline, it came nowhere near actual.
    * We can acquire a much better dataset given more time
* **Next Steps:**
    * I would like to request more time to investigate the data available on the Athena data webservice managed by the World Health Organization.
    * I also came across some similar projects that I can reference to research their findings in comparison to my own.
