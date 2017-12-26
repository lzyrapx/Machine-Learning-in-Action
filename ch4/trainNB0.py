# coding:utf-8

from numpy import *
from math import log
import numpy as np
import operator
from os import listdir


def loadDataSet():
    postingList = [['my','dog','has','flea','problem','help','please',],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stdpid']]
    classVec = [0,1,0,1,0,1] # 1 代表侮辱性文字 0代表正常言论
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 合并两个集合
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) # 创建一个其中的全部元素都为 0 的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print " %s不在字典集内!" % word
    return returnVec

# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 对每个元素做除法
    p1Vect = log(p1Num/p1Denom)  # change to log
    p0Vect = log(p0Num/p0Denom)  # change to log

    return p0Vect,p1Vect,pAbusive

if __name__ =='__main__':
    listPosts,listClass = loadDataSet()
    myVocaburaryList = createVocabList(listPosts)
    print sorted(myVocaburaryList)
    print setOfWords2Vec(myVocaburaryList,listPosts[0])
