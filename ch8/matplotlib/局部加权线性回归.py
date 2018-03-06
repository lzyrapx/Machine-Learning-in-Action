#coding: utf-8
from numpy import *

#==================局部加权线性回归================
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

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))   #产生对角线矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        #更新权重值，以指数级递减
        weights[j,j] = exp(diffMat * diffMat.T /(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "this matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] =lwlr(testArr[i],xArr,yArr,k)
    return yHat


xArr,yArr = loadDataSet('ex0.txt')
print "k=1.0：",lwlr(xArr[0],xArr,yArr,1.0) # 欠拟合
print "k=0.01：",lwlr(xArr[0],xArr,yArr,0.001)
print "k=0.003：",lwlr(xArr[0],xArr,yArr,0.003) # 过拟合

#画图
def showlwlr():
    yHat = lwlrTest(xArr, xArr, yArr, 0.01) #改参
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    import matplotlib.pyplot as plt
    fig = plt.figure() #创建绘图对象
    ax = fig.add_subplot(111)  #111表示将画布划分为1行2列选择使用从上到下第一块
    ax.plot(xSort[:,1],yHat[srtInd])
    #scatter绘制散点图
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T[:,0].flatten().A[0],s=2,c='red')
    plt.show()

showlwlr()