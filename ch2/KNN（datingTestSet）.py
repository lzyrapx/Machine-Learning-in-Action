# coding:utf-8

from numpy import *
import operator
from os import listdir


# 用于分类的输入向量inx，输入的训练样本集是dataSet，标签向量是labels
# 参数k表示用于选择最近邻居的数目

#通过KNN进行分类
def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]
    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #行向量分别相加，从而得到新的一个行向量
    distances = sqDistances**0.5

    # 对距离进行排序
    sortedDistIndicies = distances.argsort() #argsort()根据元素的值从大到小对元素进行排序，返回下标
    classCount={}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    #逆序，选取出现的类别次数最多的类别
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #返回出现的类别中次数最多的类别
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


def autoNorm(dataSet):  # 归一化特征值

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def datingClassTest():
    # 注意：一共有1000个数据
    hoRatio = 0.10  # 随机选出 10% 的数据, 对于已有的数据，将90%作为训练，剩下10%作为测试
    datingDataMat, datingLabels = filematrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) ##测试量
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前10%行的数据作为测试集，并且对测试集中的每一行都进行预测，对比测试集中实际的label
        # 后90%行的数据全部作为训练集，每个测试集样本都要跟90%的训练集计算距离，算出最相似的label
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "分类器返回为: %d, 答案为: %d" % (classifierResult, datingLabels[i])

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            print "error"

    print "测试数据量为：%d" % numTestVecs
    print "错误率: %f" % (errorCount / float(numTestVecs))
    print "错误个数：%d" % errorCount


if __name__ == '__main__':
    # print  createDataSet()
    datingClassTest()

