# Data Analysis library
import pandas as pd
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

X_train, X_validate, X_test, y_train, y_validate, y_test = pr.get_modeling_data()

def run_decision_tree(cv:int=3):
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
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print(f'Decision Tree Parameters:  {grid_DTC.best_params_}')
    # calculate scores
    train_score = grid_DTC.best_estimator_.score(X_train, y_train)
    validate_score = grid_DTC.best_estimator_.score(X_validate, y_validate)
    # save the scores into a dataframe
    scores.loc[len(scores)] = ['Decision Tree', train_score, validate_score, train_score - validate_score]



def run_random_forest(cv:int=3):
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



#################








