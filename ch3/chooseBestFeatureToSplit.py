# coding:utf-8

from numpy import *
from math import log
import numpy as np
import operator
from os import listdir

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def calcShannonEnt(dataSet): #计算熵

    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys(): #键值不存在就加入
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1 #计数

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2) # 以2为底求对数

    return shannonEnt

#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):#传递参数：待划分的数据集，划分数据集的特征(第axis个特征)，特征的返回值

    retDataSet = [] #创建新的list对象

    for featVec in dataSet: #抽取
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
           # print featVec
            retDataSet.append(reducedFeatVec)

    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):

    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet) #整个数据集的原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):        #遍历全部特征
        featList = [example[i] for example in dataSet]#创建一个新的list对象
        uniqueVals = set(featList)       #容器set
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals: #计算每一种划分方式的信息熵

            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率：特征值划分出子集概率
            prob = len(subDataSet)/float(len(dataSet))
            #因为我们在根据一个特征计算香农熵的时候，该特征的分类值是相同，这个特征这个分类的香农熵为0，
            # 即当我们的分类只有一类是香农熵是0,而分类越多，香农熵会越大
            #所以计算新的香农熵的时候使用的是子集
            newEntropy += prob * calcShannonEnt(subDataSet) #计算新的熵

        infoGain = baseEntropy - newEntropy

        if (infoGain > bestInfoGain): #计算最好的信息增量
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                      #返回一个整数，返回最好的axis

if __name__ =='__main__':

  # print createDataSet()
   dataSet ,labels = createDataSet()
   print calcShannonEnt(dataSet)

   print splitDataSet(dataSet,0,1)
   print splitDataSet(dataSet,1,1)
   print chooseBestFeatureToSplit(dataSet) # 0：The best axis
