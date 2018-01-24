# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:01:41 2017

@author: pengbo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, recall_score, classification_report


path = 'C:/Users/pengb/Desktop/abc/creditcard.csv'
data = pd.read_csv(path)

count_class = pd.value_counts(data['Class'], sort=True)

count_class.plot(kind='bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#数值大，在机器学习中可能会被增加权重，可以采取标准化
data['norm_amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
X = data.ix[:, data.columns!='Class']
y = data.ix[:, data.columns=='Class']

#分别取出正常与欺诈交易索引
number_records_fraud = len(data[data.Class==1])
fraud_indices = np.array(data[data.Class==1].index)
normal_indices = data[data.Class==0].index

#根据欺诈交易量从正常交易中随机取值
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)

#下采样
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
under_sample_data = data.iloc[under_sample_indices, :]

#在样本中分类数据
X_undersample = under_sample_data.ix[:, under_sample_data.columns!='Class']
y_undersanple = under_sample_data.ix[:, under_sample_data.columns=='Class']

print("--------------------------------------------------------------------")
print("正常交易百分比：", len(under_sample_data[under_sample_data.Class==0])/len(under_sample_data))
print("欺诈交易百分比：", len(under_sample_data[under_sample_data.Class==1])/len(under_sample_data))
print("重取样交易数量为：", len(under_sample_data))

#交叉验证，训练集中的切分训练与验证
#交叉验证目的求稳，平均值当作为模型效果
#确定随机样本一支
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("--------------------------------------------------------------------")
print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))
print("总数量：", len(X_train+X_test))

#下采样数据集
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(\
                                                                                                    X_undersample,
                                                                                                    y_undersanple,
                                                                                                    test_size=0.2,
                                                                                                    random_state=0)
print("--------------------------------------------------------------------")
print("下采样训练集数量：", len(X_train_undersample))
print("下采样测试集数量：", len(X_test_undersample))
print("下采样总数量：", len(X_train_undersample+X_test_undersample))

print("--------------------------------------------------------------------")
"""
    使用recall率检测模型
    recall rate = TP / (TP + FN)
    TP = true positive(正确的判定数量)
    FN = false positive(失败的判定数量)
"""
def printing_Kfold_scores(x_train_data, y_train_data):
    #fold=5的KFlod交叉验证
    fold = KFold(len(y_train_data), 5, shuffle=False)
    #定义不同的C参数
    c_param_range = [0.01, 0.1, 1, 10, 100]
    
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean_recall_score'])
    results_table['C_parameter'] = c_param_range
    #KFold将给出连个列表，分别是训练集索引与测试集索引
    j = 0
    for c_param in c_param_range:
        print("--------------------------------------------")
        print('C parameter:', c_param)
        print("--------------------------------------------")
        
        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            #指定C参数值并调用逻辑回归模型]
            lr = LogisticRegression(C = c_param, penalty='l1') #|w|
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0],:].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ':recall score = ', recall_acc)
        #将每个C_parameter交叉验证的recall——rate差存储
        results_table.ix[j, 'Mean_recall_score'] = np.mean(recall_accs)
        print(results_table)
        j+=1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')
    #取得最大recall——rate
    print('***************************************************************************')
    best_c = results_table.loc[results_table['Mean_recall_score'].idxmax()]['C_parameter']
    print("交叉验证的最佳模型C_parameter是：", best_c)
    print('***************************************************************************')
    return best_c

best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

def plot_confusion_matrix(cm, classes, title='Confusion matix', cmap=plt.cm.Blues):
    """
        输出混淆矩阵图例
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #设置颜色柱
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    #紧凑显示
    plt.tight_layout()
    plt.ylabel('Ture label')
    plt.xlabel('predicted lable')
    
#逻辑回归验证测试集
lr = LogisticRegression(C=best_c, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample.values)

#混淆矩阵
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)
print("Recall rate在训练数据集的矩阵：", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

#测试原始数据集
lr = LogisticRegression(C=best_c, penalty='l1')
#recall率高但是精度低
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)
cnf_Matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print("Recall rate在测试数据集的矩阵：", cnf_Matrix[1, 1]/(cnf_Matrix[1, 0]+cnf_Matrix[1, 1]))
plt.figure()
plot_confusion_matrix(cnf_Matrix, classes=class_names, title='Confusion matrix')

#不用下采样，回到原始数据训练与测试集，效果不佳
#best_c = printing_Kfold_scores(X_train, y_train)

#不同阈值下混淆矩阵效果图
lr = LogisticRegression(C=0.01, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())

#求出预测概率值
#不同阈值下的精度与recall rate（查全率）
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)
thresholds = np.arange(0.1, 1, 0.1)
plt.figure(figsize=(12, 12))
j=1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
    plt.subplot(3, 3, j)
    j += 1
    #计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)
    print("Recall rate: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' %i)

#输出分类报告
print(classification_report(y_test, y_pred))