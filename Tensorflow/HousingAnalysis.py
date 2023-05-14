# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:26:55 2023

@author: Georg
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix

df = pd.read_csv('HousingData.csv', index_col = 0)

df.hist(bins = 50, figsize = (20, 15))
plt.show()

#Categorising the income into groups so that we can take a straified sample

df['income_cat'] = pd.cut(
    df["median_income"],
    bins = [0., 1.5, 3., 4.5, 6., np.inf],
    labels = [1,2,3,4,5])

#passing this to the below, we can be sure that the proportions of people in 
#the income groups passed to the model will be the same
split = StratifiedShuffleSplit(
    n_splits = 1,
    test_size = 0.2,
    random_state = 42)

for train_index, test_index in split.split(df, df['income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis = 1, inplace = True)

#taking a copy of the data so we cna review it without breaking anything
housing = strat_train_set.copy()

#plotting the houses by location
housing.plot(
    kind = 'scatter', 
    x = 'longitude', 
    y = 'latitude',
    alpha = 0.4,
    s = housing['population']/100,
    label = 'population',
    figsize = (10,7),
    c = 'median_house_value',
    cmap = plt.get_cmap('hot'),
    colorbar = True)

plt.legend()

#checking for correlations between variables
#%%

corr_matrix = housing.corr(numeric_only = True)

attributes = [
    'median_house_value',
    'median_income',
    'total_rooms',
    'housing_median_age']

scatter_matrix(housing[attributes], figsize = (12, 8))

housing.plot(
    kind = 'scatter',
    x = 'median_income',
    y = 'median_house_value',
    alpha = 0.1)

#creating features to learn from 

housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_houshold'] = housing['population']/housing['households']

corr_matrix = housing.corr()

#%%
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

imputer = SimpleImputer

imputer = SimpleImputer(Strategy = 'Median')

