# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:26:55 2023

@author: Georg
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('HousingData.csv', index_col = 0)

df.hist(bins = 50, figsize = (20, 15))
plt.show()

