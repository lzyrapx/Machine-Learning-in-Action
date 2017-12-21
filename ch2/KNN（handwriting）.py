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

#将图像转换成向量
def img2vector(filename):

    # 创建1 * 1024的Numpy数组
    returnVect = zeros((1,1024))
    fr = open(filename)
    #文件钱32行
    for i in range(32):
        lineStr = fr.readline()
        # 每行的头32个字符
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return  returnVect

def handwritingClassTest():

    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #获取目录内容
    m = len(trainingFileList)

    trainingMat = zeros((m,1024))

    for i in range(m):
        #从文件名解析分类数据
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    #---------------------------------------------------------------#

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):

        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print "分类器返回: %d, 真正答案是: %d" % (classifierResult, classNumStr)

        if (classifierResult != classNumStr):
            errorCount += 1.0
            print "error"

    print "\n错误个数有: %d" % errorCount
    print "\n错误率: %f %%" % (errorCount/float(mTest)*100)


if __name__ == '__main__':

   testVector = img2vector("0_0.txt")
   print testVector[0,0:31]

   handwritingClassTest()

