#coding:utf-8

from numpy import *
import operator
from os import listdir


# 用于分类的输入向量inx，输入的训练样本集是dataSet，标签向量是labels
#参数k表示用于选择最近邻居的数目
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
    
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    print classify0([0][0], group,labels,3) # B
   # print classify0(([1][0],group,labels,3)) #A
    return group, labels

if __name__ == '__main__':
    print  createDataSet()
