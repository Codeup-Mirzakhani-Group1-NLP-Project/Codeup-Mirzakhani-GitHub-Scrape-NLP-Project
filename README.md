# <b><i><font size="20">Final Report Team Zillow Project</font></i></b>
## Codeup-Mirzakhani-GitHub-Scrape-NLP-Project
This is the Mirzakhani Group-1 Repository for the GitHub Scraping NLP Project

## Group 1
|Team Member |[LinkedIn](https://www.linkedin.com/school/codeup/) |[GitHub](https://github.com/Codeup-Mirzakhani-Group1-NLP-Project/Codeup-Mirzakhani-GitHub-Scrape-NLP-Project)|
|:-------------------|:---------------------------------------------|:--------------------------------------|
|Justin Evans        |linkedin.com/in/qmcbt                         |https://github.com/QMCBT-JustinEvans   |
|Nadia Paz           |linkedin.com/in/nadiapaz                      |https://github.com/nadia-paz           |
|Allante Staten      |linkedin.com/in/allantestaten                 |https://github.com/allantestaten       |
|Zachary Stufflebeme |linkedin.com/in/zachary-stufflebeme-63379a243 |https://github.com/Zachary-Stufflebeme |

# Project Overview:
What is driving the errors in the Zestimates?

This team has been tasked to collect, clean and alayze Zillow data from 2017 in order to improve a previous prediction model that was designed to predict the Home Sale Value for Single Family Properties based on available realestate data.

# Project Goal:
* Use clusters to assist in our exploration, understanding, and modeling of the Zillow data, with a target variable of logerror for this regression project.
* Discover key attributes that drive error in Zestimate logerror.
* Use those attributes to develop a machine learning model to predict impact on logerror.

# Reproduction of this Data:
* Can be accomplished using a local env.py containing user, password, host information for access to the Codeup SQL database server.
* All other step by step instructions can be found by reading and running the below Jupyter Notebook file located in our Codeup-Justin-Evans-Yvette-Ibarra github repository found here: (https://github.com/Codeup-Justin-Evans-Yvette-Ibarra/project_zillow_team).
    * Final_Report_Zillow_Team_Project.ipynb
    * wrangle.py
    * explore.py
    * model.py
    
# Initial Thoughts
Our initial thoughts are that outliers, age, and location are drivers of errors in Zestimate.

# The Plan
* Acquire data from Codeup database
* Prepare data
* Explore data in search of drivers of logerror
* Answer the following initial question:

    * **Question 1.** Do Longitude and Lattitude have a relationship with eachother and our target feature of Zillow Zestimate logerror? 

    * **Question 2.** Is there a relationship between our loc_clusters feature, and each of the five independent clusters 0-4 as binary categorical features, with logerror? 

    * **Question 3.** Is there a relationship between log_error and tax delinquency? 

    * **Question 4.** Do homes that are younger than 81 years have more log error?

    * **Question 5.** Do the clusters have a relationship with logerror and squarefeet of the home?

* Develop a Model to predict error in zestimate.
    * Use drivers identified in explore to build predictive models of error using...
    * Evaluate models on train and validate data using RMSE (Root mean square Error)
    * Select the best model based on the least RMSE
    * Evaluate the best model on test data
* Draw conclusions


# Data Dictionary:

    
## Continuous Categorical Counts
|Feature    |Description       |
|:----------|:-----------------|
|parcelid|Unique Property Index| 
|bedrooms|Number of bedrooms in home|
|bathrooms|Number of bathrooms in home including fractional bathrooms| 
|calculatedbathnbr|Continuous float64 count of Bathrooms (including half and 3/4 baths)| 
|fullbathcnt|Count of only Full Bathrooms (no half or 3/4 baths)|
|age|The age of the home in 2017| 
|yearbuilt|The Year the principal residence was built| 

## Categorical Binary
|Feature    |Description       |
|:----------|:-----------------|
|has_basement|Basement on property (if any = 1)| 
|has_deck|Deck on property (if any = 1)| 
|has_fireplace|Fireplace on property (if any = 1)| 
|has_garage|Garage on property (if any = 1)| 
|has_hottuborspa|Hot Tub or Spa on property (if any = 1)| 
|has_pool|Pool on property (if any = 1)| 
|optional_features|Property has at least one optional feature listed above (if any = 1)| 
|has_tax_delinquency|Property has had Tax Delinquncy (if any = 1)| 

## Location
|Feature    |Description       |
|:----------|:-----------------|
|fips|Federal Information Processing Standards (FIPS), now known as Federal Information Processing Series, are numeric codes assigned by the National Institute of Standards and Technology (NIST). Typically, FIPS codes deal with US states and counties. US states are identified by a 2-digit number, while US counties are identified by a 3-digit number. For example, a FIPS code of 06111, represents California -06 and Ventura County -111.|
|state|This is the two letter abbreviation for the State as defined by the FIPS code| 
|county|FIPS code for california counties|
|la_county|fips: 6037; Categorical Binary Feature for Los Angeles County (if True = 1)| 
|orange_county|fips: 6059; Categorical Binary Feature for Orange County (if True = 1)| 
|ventura_county|fips: 6111; Categorical Binary Feature for Los Angeles County (if True = 1)|
|longitude|Longitude is a measurement of location east or west of the prime meridian at Greenwich, London, England, the specially designated imaginary north-south line that passes through both geographic poles and Greenwich. Longitude is measured 180° both east and west of the prime meridian.| 
|latitude|Latitude is a measurement on a globe or map of location north or south of the Equator. Technically, there are different kinds of latitude, which are geocentric, astronomical, and geographic (or geodetic), but there are only minor differences between them.|
|zipcode|A group of five or nine numbers that are added to a postal address to assist the sorting of mail.| 
|regionidcounty|Location code that identifies the Region and County of the property within the state| 
|rawcensustractandblock|Raw unformatted data that identifies Census tracts and blocks| 
|censustractandblock|Census tracts are small, relatively permanent geographic entities within counties and Block numbering areas (BNAs) are geographic entities similar to census tracts, and delineated in counties (or the statistical equivalents of counties) without census tracts.| 

## Size
|Feature    |Description       |
|:----------|:-----------------|
|sqft|Calculated total finished living area of the home|
|lotsizesquarefeet|Calculated area of land lot belonging to parcel| 

## Value
|Feature    |Description       |
|:----------|:-----------------|
|tax_value_bldg|The total tax assessed value of the structure|
|tax_value|The total tax assessed value of the parcel| 
|tax_value_land|The total tax assessed value of the land|
|taxamount|The total tax fee to be collected on the parcel| 

## Target
|Feature    |Description       |
|:----------|:-----------------|
|log_error|This is the logerror of the Zillow Zestimate|

## Clusters
|Feature    |Description       |
|:----------|:-----------------|
|loc_clusters|Created using 'longitude', 'latitude', 'age' with n_clusters = 4|
|cluster_price_size|Created using 'taxamount', 'sqft', 'lot_sqft' with n_clusters = 4|
|cluster_delinquency_value|Created using ‘tax_value’, ‘sqft’,‘has_taxdelinquency’ with n_clusters = 4|

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

* **SPLIT:** train, validate and test (approx. 60/20/20), stratifying on target of 'churn'
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
