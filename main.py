import numpy as np
import pandas as pd
import time
import operator
import os
import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from matplotlib import pyplot as plt

#处理数据集
def data_load():
    voice_data = pd.read_csv('voice.csv')
    x = voice_data.iloc[:,:-1]
    y = voice_data.iloc[:,-1]
    y = LabelEncoder().fit_transform(y)
    imp = SimpleImputer(missing_values=0, strategy='mean')
    x = imp.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    scaler1 = StandardScaler()
    scaler1.fit(x_train)
    x_train = scaler1.transform(x_train)
    x_test = scaler1.transform(x_test)
    return x_train, x_test, y_train, y_test

class NaiveBayes:
    def __init__(self):
        self.model = None

    #求数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    #求标准差
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    #求高斯分布概率密度
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    #求和
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    #构建字典并准备计算概率需要的参数
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'GaussianNB train done!'

    #计算概率
    def cal_probability(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    #预测类别
    def predict(self, X_test):
        label = sorted(self.cal_probability(X_test).items(),key=lambda x: x[-1])[-1][0]
        return label

    #统计正确率
    def count(self, X_test, y_test):
        right = 0
        m_right = 0
        f_right = 0
        m_num = 0
        f_num = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if y == 1:
                m_num += 1
                if label == y:
                    right += 1
                    m_right += 1
            if y == 0:
                f_num += 1
                if label == y:
                    right += 1
                    f_right += 1
        return right / float(len(X_test)), m_right / float(m_num), f_right / float(f_num)

#m_rate1 = []
#f_rate1 = []
#list1 = list(range(20))
#for i in range(20):
    #X_train, X_test, y_train, y_test=data_load()
    #model = NaiveBayes()
    #model.fit(X_train, y_train)
    #rate, m_rate, f_rate=model.count(X_test, y_test)
    #print("男声正确率:", m_rate)
    #print("男声错误率", 1-m_rate)
    #print("女声正确率:", f_rate)
    #print("女声错误率", 1-f_rate)
    #m_rate1.append(m_rate)
    #f_rate1.append(f_rate)
#print("男声正确率:", m_rate1)
#print("女声正确率:", f_rate1)
#plt.figure()
#plt.plot(list1,m_rate1,color='skyblue',label='m_rate')
#plt.plot(list1,f_rate1, color='red',label='f_rate')
#plt.ylabel('right_tate')
#plt.xlabel('i')
#plt.legend()
#plt.show()
X_train, X_test, y_train, y_test=data_load()
model = NaiveBayes()
model.fit(X_train, y_train)
rate, m_rate, f_rate=model.count(X_test, y_test)
print("男声正确率:", m_rate)
print("男声错误率", 1-m_rate)
print("女声正确率:", f_rate)
print("女声错误率", 1-f_rate)