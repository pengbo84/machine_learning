# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1 12:23:50 2017

@author: pengbo
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import KFold


path = 'C:/Users/pengb/Desktop/abc/titanic_train.csv'
titanic = pd.read_csv(path)

#用均值填充缺失值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
#性别0-1转换
titanic.loc[titanic['Sex'] == 'male', 'Sex']=0
titanic.loc[titanic['Sex'] == 'female', 'Sex']=1

#上船地点转换
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] =0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] =1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] =2

#预测Survived
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#初始化算法类别
alg = LinearRegression()
#非随机交叉验证
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    #训练算法
    train_target = titanic['Survived'].iloc[train]
    #用predictors与目标集训练算法
    alg.fit(train_predictors, train_target)
    #预测
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

#形成一个整体
predictions = np.concatenate(predictions, axis=0)

predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print('预测准确率为：', accuracy)

#逻辑回归预测
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], 
                                          titanic['Survived'], cv=3)
print('预测准确率为：', scores.mean())