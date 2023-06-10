# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:12:57 2023

@author: Georg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

def plotVals(df, y):
    fig, ax = plt.subplots()
    sns.boxplot(data = df, y = y, x = 'Survived', hue = 'Sex')

def cleanDf(df):
    """Preprocessing step"""
    sexMap = {'male': 0, 'female': 1}
    df['Sex'] = df['Sex'].map(sexMap)
    return df

def trainOnAge(df, Model, cols):
    """Trains the model 'Model' on the ages in the passed dataframe
    using the information in cols"""
    train_df = df[df['Age'] > 0]
    X_train = train_df[cols]
    y_train = train_df['Age']
    Model.fit(X_train, y_train)
    scores = cross_val_score(Model, X_train, y_train, 
                             scoring = "neg_mean_squared_error",
                             cv = 10)
    tree_rmse_scores = np.sqrt(-scores)
    print("Results for the Age Regressor")
    print("Mean Score: {}".format(tree_rmse_scores.mean()))
    print("Standard Deviation: {} \n".format(tree_rmse_scores.std()))
    
def addAge(df, Model, cols):
    """Adds values to the age column if they are missing by making
    preicitons with the passed model"""
    temp = []
    dataDict = {}
    for i in range(len(df['Age'])):
        if df['Age'][i] >= 0:
            temp.append(df['Age'][i])
        else:
            for col in cols:
                dataDict[col] = df[col].iloc[i]
            data = pd.DataFrame(dataDict,index = [0])
            temp.append(int(Model.predict(data)))
    df['Age2'] = temp
    
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(df['Age'].values.reshape(-1, 1))
    df['Age'] = imputer.transform(df['Age'].values.reshape(-1, 1))
    return df

def trainOnSurvived(X_train, y_train, Model, param_grid):
    grid_search = GridSearchCV(Model, param_grid, cv = 5,
                               scoring = 'roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
    
def predictSurvived(X, Model, y = [], method = '', ax = ''):
    if len(y) > 0:
        Model.predict(X)
        cvs = cross_val_score(Model, X, y, cv = 5, scoring = 'accuracy')
        print("Results for the Survived Model Fitting")
        print("Accuracy: {}%".format(round(cvs.mean(), 5)))
        
        cvs_pred = cross_val_predict(Model, X, y, cv = 5, 
                                     method = method)
        #plot roc_curv
        #set up graph
        ax.plot([0,1], [0,1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if method == 'decision_function':
            fpr, tpr, thresholds = roc_curve(y, cvs_pred)
            ax.plot(fpr, tpr, label = str(Model))
        elif method == 'predict_proba':
            forest_scores = cvs_pred[:, 1]
            fpr, tpr, thresholds = roc_curve(y, forest_scores)
            ax.plot(fpr, tpr, label = str(Model))

    else:
        Model.predict(X)

def makePredictions(df, cols, Model):
    """Make Prdictions and submit to Kaggle"""
    df = cleanDf(df)
    survival_predictions = Model.predict(df[cols])
    survival_predictions = survival_predictions.astype(int)

    output = pd.DataFrame({'PassengerId': df.PassengerId, 
                           'Survived': survival_predictions})
    output.to_csv('submission.csv', index=False)
    os.system('cmd /k "kaggle competitions submit -c titanic -f submission.csv -m "Latest test"')

def saveChangesToRepo():
    os.system(r'cmd /c "cd C:\Users\Georg\MLPractice"')
    os.system('cmd /c "git add ."')
    os.system('cmd /c "git commit -m "Making changes to Titanic"')
    os.system('cmd /c "git push"')
                

#Cleaning data before training model
train_df = pd.read_csv('train.csv')
train_df = cleanDf(train_df)
DTR = DecisionTreeRegressor()
ageCols = ['Parch', 'SibSp', 'Fare']
trainOnAge(train_df, DTR, ageCols)
addAge(train_df, DTR, ageCols)

#Training Model
cols = ['Parch', 'Pclass', 'SibSp', 'Sex']
X_train = train_df[cols]
y_train = train_df['Survived']
fig, ax = plt.subplots()

RFC = RandomForestClassifier(random_state = 42)
param_grid = [{'n_estimators' : [100, 200], 
               'criterion' : ['gini', 'entropy'],
               'bootstrap' : [True]}]
RFC = trainOnSurvived(X_train, y_train, RFC, param_grid)
predictSurvived(X_train, RFC, y_train, method = 'predict_proba', ax = ax)

param_grid = [{'n_estimators' : [500, 1000],
               'max_samples' : [100],
               'bootstrap' : [True],
               'n_jobs' : [-1]}]
bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state = 42))
bag_clf = trainOnSurvived(X_train, y_train, bag_clf, param_grid)
predictSurvived(X_train, bag_clf, y_train, method = 'predict_proba', ax = ax)

#Make Predicitions based on test data

#makePredictions(pd.read_csv('test.csv'), cols, bag_clf)

saveChangesToRepo()







    

