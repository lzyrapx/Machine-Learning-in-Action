#coding:utf-8

from numpy import *

def loadDataSet():
    dataMat = [] #list
    labelMat = [] #list
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(z):
    return 1.0 / (1 + exp(-z))

# Logistic 回归梯度上升优化算法
def gradAscent(datamat,classlabel):
    dataMat = mat(datamat) #convert to NumPy matrix
    labeMat = mat(classlabel).transpose() #convert to NumPy matrix
    m,n = shape(dataMat)
    alpha = 0.001 # 向目标移动的步长
    maxCycles = 500 # 迭代次数
    weight = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weight)
        diff = labeMat - h #误差
        weight += alpha * dataMat.transpose()*diff #这里不止一次乘积运算
    return weight

def stocGradAscent(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2] #最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == "__main__" :
    dataMat,labelMat = loadDataSet()
    # print dataMat,labelMat
    weight = stocGradAscent(array(dataMat),labelMat)
    plotBestFit(weight)


