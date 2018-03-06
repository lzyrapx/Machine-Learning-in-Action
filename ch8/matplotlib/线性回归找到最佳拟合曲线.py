#coding: utf-8
from numpy import *

# ===========用线性回归找到最佳拟合曲线===========
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


# 测试上边的函数
xArr, yArr = loadDataSet("ex0.txt")
# xArr, yArr = loadDataSet("ex1.txt")
ws = standRegress(xArr, yArr)
print "ws（相关系数）：\n", ws  # ws 存放的就是回归系数

def show():
    import matplotlib.pyplot as plt
    xMat = mat(xArr);
    yMat = mat(yArr)
    yHat = xMat * ws
    fig = plt.figure()  # 创建绘图对象
    ax = fig.add_subplot(111)  # 111表示将画布划分为1行2列选择使用从上到下第一块
    # scatter绘制散点图
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # 复制，排序
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    # plot画线
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


show()

yHat = mat(xArr) * ws
# yHat = xMat * ws
# 利用numpy库提供的corrcoef来计算预测值和真实值得相关性
print "相关性：\n", corrcoef(yHat.T, mat(yArr))