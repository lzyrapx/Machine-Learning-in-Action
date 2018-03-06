#coding: utf-8
from numpy import *

# 加载数据集
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat

# 计算最佳拟合曲线
def standRegress(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T  # .T代表转置矩阵
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # linalg.det(xTx) 计算行列式的值
        print "This matrix is singular , cannot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

#==========前向逐步回归============

#计算平方误差
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

#数据标准化处理
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1));
    wsTest = ws.copy();
    wsMax = ws.copy()
    for i in range(numIt): #could change this to while loop
        #print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


xArr,yArr = loadDataSet('abalone.txt')

# 把这些结果与最小二乘法进行比较，后者的结果可以通过如下代码:

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regularize(xMat)
yM = mean(yMat,0)
yMat = yMat - yM
weights = standRegress(xMat, yMat.T)
print weights.T

# print stageWise(xArr, yArr, 0.01, 200)
mat = stageWise(xArr,yArr,0.005,1000) # 使用0.005的epsilon 迭代 1000次

def showRidge():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mat)
    plt.show()
showRidge()


