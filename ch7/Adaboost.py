#coding:utf-8

from numpy import *
def loadSimpData():
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3 , 1.],
                     [1. , 1.],
                     [2. ,1.]])
    classLabels = [1.0 , 1.0 , -1.0 ,-1.0 ,1.0]
    return datMat , classLabels

#自适应数据加载函数
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#单层决策树生成函数
#通过阈值比较对数据进行分类函数，在阈值一边的会分到-1类别，另一边的分到类别+1
#先全部初始化为1，然后进行过滤，不满足不等式的变为-1
def stumpClassify(dataMatrix , dimen , threshVal , threshIneq) :
    retArray = ones((shape(dataMatrix)[0] , 1 ))
    if threshIneq == 'lt' :
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#遍历上述函数所有可能输入，找到最佳单层决策树
# buildStump函数得到构建AdaBoost算法需要所有信息，字典、错误率、类别估计值等
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr) ; labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSetps = 10.0 #在特征的所有可能值上进行遍历
    bestStump = {}  #存储给定权重D得到的最佳单层决策树
    bestClasEst = mat(zeros((m,1)))
    minError = inf #初始化为无穷大，找最小错误率
    for i in range(n) :#在特征上进行遍历，计算最大最小值来求得合理步长
        rangeMin = dataMatrix[:,i].min() ; rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSetps
        for j in range(-1,int(numSetps)+1):
            for inequal in ['lt' , 'gt'] :#大于小于切换不等式
                threshVal = (rangeMin+float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1))) #如果预测值≠真实值，为1
                errArr[predictedVals==labelMat] = 0
                weightedError = D.T * errArr #相乘求和得到错误权重数值
                # print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f " % (
                # i, threshVal, inequal, weightedError)
                if weightedError < minError :
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq']  = inequal
    return bestStump , minError , bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt = 40) : #=数据集，类别标签，迭代次数numIt
    weakClassArr = []
    m = shape(dataArr)[0] #m是数据的数目
    D = mat(ones((m,1))/m) #每个数据点的权重
    aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值
    for i in  range(numIt): #如果在迭代次数内错误率为0则退出
        bestStump , error , classEst = buildStump(dataArr,classLabels,D)
        #返回利用D得到的最小错误率单层决策树，最小的错误率和估计的类别向量
        #print "D:" , D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #分类器分配的权重，这里比较是为了防止0出现溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print "classEst : " , classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T , classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        #print "aggClassEst : " , aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T , ones((m,1)))
        errorRate = aggErrors.sum() / m
        print "Total error : " , errorRate , "\n"
        if errorRate ==0.0 : break #错误率等于0就 break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass) # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

#ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    print "the Area Under the Curve is: ", ySum * xStep
    plt.show()


if __name__ == "__main__":
    # datMat, classLabels = loadSimpData()
    # D = mat(ones((5, 1)) / 5)
    # print buildStump(datMat, classLabels, D)

    # -----------------------------------------------------------

    # classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    # print classifierArray

    # -----------------------------------------------------------

    # datArr, labelArr = loadSimpData()
    # classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
    # print adaClassify([0, 0], classifierArr)
    # print ""
    # print adaClassify([[5, 5], [0, 0]], classifierArr)

    # -----------------------------------------------------------
    # datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    # classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
    #
    # # print classifierArray
    # testArr, testLabelArr = loadDataSet("horseColicTest2.txt")
    # prediction10 = adaClassify(testArr, classifierArray)
    # errArr = mat(ones((67, 1)))
    # print errArr[prediction10 != mat(testLabelArr).T].sum()

    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classfierArray,aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    plotROC(aggClassEst.T,labelArr)
