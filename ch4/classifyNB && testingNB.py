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
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
   # p1Vect = log(p1Num/p1Denom)  # change to log
   # p0Vect = log(p0Num/p0Denom)  # change to log

    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)


if __name__ =='__main__':
    listPosts,listClass = loadDataSet()
    myVocaburaryList = createVocabList(listPosts)
   # print sorted(myVocaburaryList)
   # print setOfWords2Vec(myVocaburaryList,listPosts[0])
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocaburaryList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClass)
    #print p0V,p1V
    print pAb
    testingNB()
