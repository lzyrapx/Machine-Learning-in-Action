# coding:utf-8

from numpy import *
import numpy as np
import operator
from os import listdir


# 用于分类的输入向量inx，输入的训练样本集是dataSet，标签向量是labels
# 参数k表示用于选择最近邻居的数目

# 通过KNN进行分类
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 行向量分别相加，从而得到新的一个行向量
    distances = sqDistances ** 0.5

    # 对距离进行排序
    sortedDistIndicies = distances.argsort()  # argsort()根据元素的值从大到小对元素进行排序，返回下标
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 逆序，选取出现的类别次数最多的类别
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回出现的类别中次数最多的类别
    return sortedClassCount[0][0]


def filematrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 文件中的行数
    returnMat = zeros((numberOfLines, 3))  # 初始化矩阵
    classLabelVector = []  # 初始化labels
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def createDataSetFromFile(filename):
    # Read lines
    file = open(filename)
    lines = file.readlines()
    file.close()

    # Change lines into array
    featureCount = len(lines[0].split()) - 1
    group = np.zeros((len(lines), featureCount))
    labels = []

    for i in range(len(lines)):
        lst = lines[i].split()
        group[i] = np.array(lst[:-1])
        labels.append(lst[-1])

    return (group, labels)


def autoNorm(dataSet):  # 归一化特征值

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def datingClassTestOne():
    # 注意：一共1000个数据
    hoRatio = 0.10  # 随机选出 10% 的数据, 对于已有的数据，将90%作为训练，剩下10%作为测试
    datingDataMat, datingLabels = filematrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):

        # 前10%行的数据作为测试集，并且对测试集中的每一行都进行预测，对比测试集中实际的label
        # 后90%行的数据全部作为训练集，每个测试集样本都要跟90%的训练集计算距离，算出最相似的label
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # print "分类器返回为: %d, 答案为: %d" % (classifierResult, datingLabels[i])

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            #    print "error"
    print "(单数据集，部分作训练，部分作测试)测试数据量为：%d" % numTestVecs
    print "正确率: %f%%" % ((numTestVecs - errorCount) / float(numTestVecs) * 100.0)
    print "正确个数：%d" % (numTestVecs - errorCount)


def datingClassTestTwo():
    ## getting training set
    TrainingGroup, TraningLabels = createDataSetFromFile("datingTraningSet.txt")
    TrainingGroup, ranges, minVals = autoNorm(TrainingGroup)  # 归一化特征值

    # try on the test set
    testGroup, testLabels = createDataSetFromFile('datingTestSet.txt')
    testGroup, ranges, minVals = autoNorm(testGroup)  # 归一化特征值

    # print "训练数据量：%d" % TrainingGroup.shape[0] #1000
    # print "测试数据量：%d" %testGroup.shape[0] #1000

    TrainingGroupSize = int(TrainingGroup.shape[0] * 0.1)
    print "(最准确)两个数据集的数据各测试数据量：%d" % TrainingGroupSize
    correct = 0
    for i in range(TrainingGroupSize):
        res = classify0(testGroup[i], TrainingGroup, TraningLabels, 3)
        if res == 'didntLike':
            res = '1'
        elif res == 'smallDoses':
            res = '2'
        else:
            res = '3'
        if res == testLabels[i]:
            correct += 1
    #print  correct
    print "正确率：%f%% " % (correct / float(TrainingGroupSize) * 100.0)
    print "正确个数：%d" % correct


if __name__ == '__main__':
    # print  createDataSet()

    datingClassTestTwo()

    print "--------------"
    print "--------------"

    datingClassTestOne()


