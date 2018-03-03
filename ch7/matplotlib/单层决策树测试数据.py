#coding: utf-8

from numpy import *
import matplotlib
import matplotlib.pyplot as plt



# def loadDataSet(fileName):
#     dataMat = []; labelMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         lineArr = line.strip().split('\t')
#         dataMat.append([float(lineArr[0]), float(lineArr[1])])
#         labelMat.append(float(lineArr[2]))
#     return dataMat,labelMat


if __name__ == "__main__" :
    # datMat,classLabels = loadDataSet("horseColicTraining2.txt")
    # datMat = matrix(datMat)
    # print datMat
    # print "size=%d" % len(datMat)
    # print datMat[1,0]
    # print classLabels
    # print "size=%d" % len(classLabels)
    datMat = matrix([[ 1. ,  2.1],
            [ 1.5,  1.6],
            [ 1.3,  1. ],
            [ 1. ,  1. ],
            [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers = []
    colors = []

    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            xcord1.append(datMat[i, 0]), ycord1.append(datMat[i, 1])
        else:
            xcord0.append(datMat[i, 0]), ycord0.append(datMat[i, 1])
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('decision stump test data')
    plt.show()
