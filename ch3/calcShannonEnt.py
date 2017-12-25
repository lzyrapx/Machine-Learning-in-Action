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


if __name__ =='__main__':
   # print createDataSet()
   dataSet ,labels = createDataSet()
   print calcShannonEnt(dataSet)
