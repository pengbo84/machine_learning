# -*- coding: utf-8 -*-
"""
Created on Thur Oct 26 12:20:55 2017

@author: pengbo
"""

#过采样
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from 样本不均衡解决方案及逻辑回归应用 import printing_Kfold_scores, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression


path = 'C:/Users/pengb/Desktop/abc/creditcard.csv'
credit_cards = pd.read_csv(path)
columns = credit_cards.columns
feature_columns = columns.delete(len(columns)-1)
features = credit_cards[feature_columns]
labels = credit_cards['Class']

features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=0)

oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(features_train, labels_train)

os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
best_c = printing_Kfold_scores(os_features, os_labels)

lr = LogisticRegression(C = best_c, penalty='l1')
lr.fit(os_features, os_labels.values.ravel())
y_pred = lr.predict(features_test.values)
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)
print("Recall rate在测试数据集的矩阵：", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')