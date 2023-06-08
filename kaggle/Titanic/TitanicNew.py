# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:12:57 2023

@author: Georg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

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
    print("Mean Score: {}".format(tree_rmse_scores.mean()))
    print("Standard Deviation: {}".format(tree_rmse_scores.std()))
    
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
    df['Age'] = temp
    return df

def trainOnSurvived():

#Cleaning data before training model
train_df = cleanDf(train_df)
DTR = DecisionTreeRegressor()
ageCols = ['Parch', 'SibSp', 'Pclass']
trainOnAge(train_df, DTR, ageCols)
addAge(train_df, DTR, ageCols)

#Training Model




    

