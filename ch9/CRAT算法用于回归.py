#coding: utf-8

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

#拆分数据集函数，二元拆分法
#@dataSet：待拆分的数据集
#@feature：作为拆分点的特征索引
#@value：特征的某一取值作为分割值
def binSplitDataSet(dataSet, feature, value):

    # 采用条件过滤的方法获取数据集每个样本目标特征的取值大于value的样本存入mat0
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]  # 书本错误 typo
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    #样本目标特征取值不大于value的样本存入mat1
    # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0] # 书本错误 typo
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

#回归树的切分函数

#叶节点生成函数
def regLeaf(dataSet):
    #数据集列表最后一列特征值的均值作为叶节点返回
    return mean(dataSet[:,-1])

#误差计算函数
def regErr(dataSet):
    #计算数据集最后一列特征值的均方差*数据集样本数，得到总方差返回
    return var(dataSet[:,-1])*shape(dataSet)[0]


def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    # 数据集最后一列所有的值都相同
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        # 最优特征返回none，将该数据集最后一列计算均值作为叶节点值返回
        return None, leafType(dataSet)

    m,n = shape(dataSet)
    # 计算未切分前数据集的误差
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    # 遍历数据集所有的特征，除最后一列目标变量值
    for featIndex in range(n-1):
        # 遍历每个特征里不同的特征值
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
        #for splitVal in set(dataSet[:,featIndex]):  # 书本错误 typo
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分后比切分前误差下降值未达到tolS
    if (S - bestS) < tolS:
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    # 返回最佳切分特征及最佳切分特征取值
    return bestIndex,bestValue

#创建树函数
#@dataSet：数据集
#@leafType：生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#@errType：计算误差的类型 1 回归错误类型：总方差=均方差*样本数 2 模型错误类型：预测误差(y-yHat)平方的累加和
#@ops：用户指定的参数
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops=(1,4)):

    #选取最佳分割特征和特征值
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    #如果特征为none，直接返回叶节点值
    if feat == None:return val
    #树的类型是字典类型
    retTree={}
    #树字典的一个元素是切分的最佳特征
    retTree['spInd']=feat
    #第二个元素是最佳特征对应的最佳切分特征值
    retTree['spVal']=val
    #根据特征索引及特征值对数据集进行二元拆分，并返回拆分的两个数据子集
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    #第三个元素是树的左分支，通过lSet子集递归生成左子树
    retTree['left']=createTree(lSet,leafType,errType,ops)
    #第四个元素是树的右分支，通过rSet子集递归生成右子树
    retTree['right']=createTree(rSet,leafType,errType,ops)
    #返回生成的数字典
    return retTree

if __name__ == "__main__" :
    myDat = loadDataSet("ex00.txt")
    myMat = mat(myDat)
    print createTree(myMat)
    # {'spInd': 0, 'spVal': 0.48813, 'right': -0.044650285714285719, 'left': 1.0180967672413792}
    myDat1 = loadDataSet("ex0.txt")
    MyMat1 = mat(myDat1)
    print createTree(MyMat1)
    #{'spInd': 1, 'spVal': 0.39435, 'right': {'spInd': 1, 'spVal': 0.197834, 'right': -0.023838155555555553, 'left': 1.0289583666666666},
    # 'left': {'spInd': 1, 'spVal': 0.582002, 'right': 1.980035071428571, 'left': {'spInd': 1, 'spVal': 0.797583, 'right': 2.9836209534883724, 'left': 3.9871631999999999}}}

