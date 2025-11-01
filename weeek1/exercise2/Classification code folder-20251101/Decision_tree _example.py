# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:58:40 2024

@author: turunenj
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

iris = load_iris()

cross_val_score(clf, iris.data, iris.target, cv=10)