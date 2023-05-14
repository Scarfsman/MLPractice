# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:20:32 2023

@author: Georg
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def trainOnAge(mod):
    mastercheck = [int('Master' in train_df['Name'][i]) for i in range(len(train_df['Name']))]
    parent = [int(train_df['Parch'][i] > 2) for i in range(len(train_df['Name']))]
    train_df['Master'] = mastercheck
    train_df['Parent'] = parent

    age_train = train_df[train_df['Age'] >= 0]
    age_x = age_train[['Master', 'Parent']]
    age_y = age_train['Age']
    mod.fit(age_x, age_y)

def replaceMissing(df, func):
    temp = []
    for i in range(len(df['Age'])):
        if df['Age'][i] >= 0:
            temp.append(df['Age'][i])
        else:
            data = pd.DataFrame({'Master': df['Master'].iloc[i],
                                 'Parent': df['Parent'].iloc[i]},
                                index = [0])
            temp.append(int(func.predict(data)))
    df['Age2'] = temp
 
def addCols(df):
    mastercheck = [int('Master' in df['Name'][i]) for i in range(len(df['Name']))]
    parent = [int(df['Parch'][i] > 2) for i in range(len(df['Name']))]
    df['Master'] = mastercheck
    df['Parent'] = parent    

np.random.seed(42)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

sexMap = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(sexMap)
test_df['Sex'] = test_df['Sex'].map(sexMap)

survive_corrs = train_df.corr()['Survived'].sort_values(ascending = False)

#Train a regressor to impute missing age values

tree_reg = DecisionTreeRegressor()
trainOnAge(tree_reg)
replaceMissing(train_df, tree_reg)
        
#Train the classifer
train_df = train_df[['Sex', 'Age', 'Age2', 'Fare', 'Survived', 'Parent']]
imputer = SimpleImputer(strategy = 'median')
imputer.fit(train_df)
cols = train_df.columns
train_df = pd.DataFrame(imputer.transform(train_df), columns = cols)

#using the DecisionTreeRegressor
x = train_df[['Sex', 'Age', 'Fare']]
y = train_df['Survived']

tree_cla = DecisionTreeClassifier()
tree_cla.fit(x, y)

forr_cla = RandomForestClassifier()
forr_cla.fit(x, y)

survival_predictions = tree_cla.predict(x)
tree_mse  = mean_squared_error(y, survival_predictions)
tree_rmse = np.sqrt(tree_mse)

param_grid = [{'n_estimators': [3, 10, 30], 
               'max_features': [2, 4, 6, 8]},
              
              {'bootstrap' : [False], 
              'n_estimators': [3, 10], 
              'max_features': [4, 8]}]

grid_search = GridSearchCV(forr_cla, param_grid, cv = 5,
                           scoring = 'neg_mean_squared_error',
                           return_train_score = True)

grid_search.fit(x, y)

final_model = grid_search.best_estimator_
final_pred = final_model.predict(x)
final_mse = mean_squared_error(y, final_pred)
final_rmse = np.sqrt(final_mse)

addCols(test_df)
replaceMissing(test_df, tree_reg)
x = test_df[['Sex', 'Age', 'Fare']]
imputer.fit(x)
cols = x.columns
x = pd.DataFrame(imputer.transform(x), columns = cols)

survival_predictions = final_model.predict(x)
survival_predictions = survival_predictions.astype(int)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': survival_predictions})
output.to_csv('submission.csv', index=False)



