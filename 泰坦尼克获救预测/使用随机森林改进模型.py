# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 16:57:00 2017

@author: pengbo
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier


path = 'C:/Users/pengb/Desktop/abc/titanic_train.csv'
titanic = pd.read_csv(path)

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
alg = RandomForestClassifier(random_state=1, n_estimators=50, 
                             min_samples_split=14,
                             min_samples_leaf=1)

kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], 
                                          titanic['Survived'], cv=kf)
print('预测准确率为：', scores.mean())

#进一步提取数据特征
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))

def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

#获取所有头衔并输出相应出现次数
titles = titanic['Name'].apply(get_title)
print(pd.value_counts(titles))
#并输出相应出现次数
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5,
                 'Rev': 6, 'Major': 7, 'Col': 7, 'Mile': 8, 'Don': 9,
                 'Lady': 10, 'Mlle': 11, 'Ms': 12, 'Sir': 13, 'Mme': 14, 
                 'Jonkheer': 15, 'Capt': 16, 'Countess': 17}
#将头衔数值化
for k, v in title_mapping.items():
    titles[titles==k] = v
print(pd.value_counts(titles)) 
#加入原始数据
titanic['Title'] = titles

#添加特征值Title
predictors = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
              'NameLength', 'FamilySize', 'Title', 'Pclass']
    
selector = SelectKBest(f_classif, k=5)  
selector.fit(titanic[predictors], titanic['Survived'])

#取得原始数据的p-values值，并将p-values值转换为scores
scores = -np.log10(selector.pvalues_)

#将scores可视化，以便查看哪一个特征对生还可能性影响最大
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
    
#只选取四个最佳特征
predictors = ['Pclass', 'Sex', 'NameLength', 'Title']
    
#构建新的算法
alg = RandomForestClassifier(random_state=1, 
                             n_estimators=50, 
                             min_samples_split=8,
                             min_samples_leaf=4)    
    
algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
        ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','FamilySize', 'Title']],
        [LogisticRegression(random_state=1), 
        ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','FamilySize', 'Title']]]  

#再次初始化交叉验证   
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)    

predictions = []
for train, test in kf:
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
 
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] =0
    test_predictions[test_predictions > .5] =1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0) 
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print('预测准确率为：', accuracy)