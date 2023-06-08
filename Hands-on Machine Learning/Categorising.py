# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:47:36 2023

@author: Georg
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version = 1, parser='auto')

#%%

def getMetrics(train, pred):
    print(confusion_matrix(train, pred))
    print(precision_score(train, pred))
    print(recall_score(train, pred))
    
def precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label = "Recall")
    plt.legend()

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
some_digit = np.array(X.iloc[0])
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = 'binary')
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 5, method = "decision_function")

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method =  "decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
precision_recall_vs_threshold(precisions, recalls, thresholds)


