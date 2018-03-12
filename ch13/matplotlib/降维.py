#coding:utf-8

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pca

dataMat = pca.loadDataSet('testSet.txt')
lowDMat, reconMat = pca.pca(dataMat, 1)
# lowDMat, reconMat = pca.pca(dataMat,2) #保留原来的2维数据，画图后可看出，数据样本是重合的

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
#由两维降为1维数据，降维后为一条红色直线，该方向是样本方差最大的方向，即样本离散程度最大的方向，该方向，将原来的2维数据融合为1维上
plt.show()
