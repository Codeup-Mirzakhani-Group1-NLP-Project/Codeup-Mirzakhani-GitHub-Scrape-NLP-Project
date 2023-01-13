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
# random state seed for split and classification algorithms
seed = 42
# DataFrame to store the scores
scores = pd.DataFrame(columns=['model_name', 'train_score', 'validate_score', 'score_difference'])

# create sets and taget variables for modeling
X_train, X_validate, X_test, y_train, y_validate, y_test = pr.get_modeling_data()

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








