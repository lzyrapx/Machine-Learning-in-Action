#coding:utf-8

import matplotlib
from numpy import *

#解析文本数据
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #将每行数据映射为浮点数
        fltLine=map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

if __name__=="__main__":
    import matplotlib.pyplot as plt
    myDat = loadDataSet("ex2.txt")
    myMat = mat(myDat)
    plt.plot(myMat[:,0],myMat[:,1],'ro')
    plt.show()

