import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing as pp


#导入数据，路径最好不要包含中文字符
path = 'C:/Users/pengb/Desktop/abc/LogiReg_data.txt'
#由于数据没有cols，所以header设置为None，同时指定columns name
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

#取出子集
positive = pdData[pdData['Admitted']==1]
negative = pdData[pdData['Admitted']==0]

fig, ax = plt.subplots(figsize=(12, 6))
#散点图展示
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=60, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

#自定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#构造200个等空间值向量
nums = np.arange(-100, 100, step=1) 
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(nums, sigmoid(nums), 'g')

#构造预测函数
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

#求偏置量权重
pdData.insert(0, 'Ones', 1)
print(pdData)
origin_data = pdData.as_matrix()
print(origin_data)
print(type(origin_data))
cols = origin_data.shape[1]
X = origin_data[:, 0:cols-1]
y = origin_data[:, cols-1:cols]
#设置参数,初始化占位
theta = np.zeros([1, 3])
print(X[:5])
print(y[:5])

#构造损失函数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))
print(cost(X, y, theta))

#求梯度
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad

#比较3种不同梯度下降方法
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    else:
        return np.linalg.norm(value) < threshold
    
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

def descent(data, theta, batchSize, stopType, thresh, alpha):
    init_time = time.time()
    i = 0 #迭代次数
    k = 0 #小批量迭代batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]
    
    while True:
        grad = gradient(X[k:k+batchSize],y[k:k+batchSize],theta) # 梯度
        k += batchSize # 取batch数量个数据
        if k>=n:
            k = 0
            X,y = shuffleData(data) # 重新洗牌
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X,y,theta)) # 计算新的损失
        i += 1
        
        if stopType == STOP_ITER:   
            value = i
        elif stopType == STOP_COST: 
            value = costs
        elif stopType == STOP_GRAD: 
            value = grad
        if stopCriterion(stopType,value,thresh):break
    return theta,i-1,costs,grad,time.time()-init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace()
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: 
        strDescType = "Gradient"
    elif batchSize==1:  
        strDescType = "Stochastic"
    else: 
        strDescType = "Mini-batch ({})".format(batchSize)
        name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: 
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: 
        strStop = "costs change < {}".format(thresh)
    else: 
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".\
           format(name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta

n = 100

#调用sk-learn进行数据预处理
scaled_data = origin_data.copy()
scaled_data[:, 1:3] = pp.scale(origin_data[:, 1:3])

theta = runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.004, alpha=0.001)

#确定使用minibatch方法
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]

scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
#sigmoid>0.5且y标签为1意味着模型预测正确
#zip将correct值与y值一一对应打包
correct = [1 if ((a ==1 and b ==1) or (a == 0 and b==0)) else\
           0 for (a, b) in zip(predictions, y)]
#使用高阶函数map将correct值转换为整数
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))





