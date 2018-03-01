#coding: utf-8

from numpy import *
import numpy as np
from time import sleep

def loadDataSet(filename): #读入数据
    dataMat = [] ; labelMat = [] #创建两个数组
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t') #对当前行进行去回车，空格操作
        dataMat.append([float(lineArr[0]),float(lineArr[1])]) #将两个特征加入dataMat
        labelMat.append((float(lineArr[2])))#将标签加入labelMat
    return dataMat,labelMat

def selectJrand(i,m):#用于在区间内选择一个整数，i为alpha的下标，m为alpha的个数
    j = i
    while(j==i):#只要函数值不等于输入值i就会随机，因为要满足 ∑alpha(i)*label(i)=0,同时改变两个alpha
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):#用来调整大于H或小于L的alpha值
    if aj>H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版SMO
# 这本数最大的一个函数
# 输入参数：数据集，类别标签，常数C，容错率，取消前最大的循环次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):#数据集，类别标签，常熟C，容错率，退出前的最大循环次数
    dataMatrix = mat(dataMatIn) ;  #转换成numpy矩阵
    labelMat = mat(classLabels).transpose() #转换成numpy矩阵
    b = 0 ; m,n = shape(dataMatrix) #求出行列
    alphas = mat(zeros((m,1)))#讲alpha都初始化为0
    iter = 0#没有任何alpha改变下的遍历数据集的次数
    while (iter < maxIter) : #当迭代次数小于最大迭代次数
        alphaPairsChanged = 0 #用来记录alpha是否被优化
        for i in range(m): #对m行数据进行处理
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b #预测的类别
            Ei = fXi - float(labelMat[i]) #误差Ei
            #如果误差很大，就可以基于该组数据所对应的alpha进行优化
            if ((labelMat[i]*Ei < -toler )and (alphas[i] < C )) or ((labelMat[i]*Ei > toler ) and alphas[i]>0 ) :
            #在if语句，测试正间隔和负间隔，同时检查alpha值,保证其不能等于0或C
                j = selectJrand(i,m) #随机第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy() #把两个alpha赋值，这样的好处是不改变原有alphas的值
                if(labelMat[i] != labelMat[j]):#如果标签向量不相等，保证alpha再0~C之间
                    L = max(0,alphas[j]+alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i] - C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H : print("L==H") ; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T #是alpha[j]的最优修改量
                if eta >= 0 : print "eta>=0";continue
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L) #调整alpha的大小
                if(abs(alphas[j]-alphaJold) < 0.00001) : print "j not moving enough " ; continue#检查alpha[j]
                alphas[i]+=labelMat[i]*labelMat[j]*(alphaJold-alphas[j]) #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T- \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]) : b = b1
                elif (0<alphas[j]) and (C>alphas[j]) : b = b2
                else: b = (b1 + b2 ) / 2.0
                alphaPairsChanged+=1
                print "iter : %d i:%d , pairs changed %d " % (iter , i , alphaPairsChanged)
        if (alphaPairsChanged==0) : iter +=1
        else:iter = 0
        print "ietration number %d " % iter
    return b,alphas


if __name__ == "__main__":
    dataMat,labelMat = loadDataSet("testSet.txt")
    # print labelMat
    print smoSimple(dataMat,labelMat,0.6,0.001,40)
