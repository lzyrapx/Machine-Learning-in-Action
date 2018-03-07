#coding:utf-8

from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = map(float,curline)
        dataMat.append(fltline)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet , k):#K个随机质心
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n))) #k个
    for j in range(n):
        minJ = min(dataSet[:,j]) #找到边界最小值
        rangeJ = float(max(dataSet[:,j])-minJ)#最大减最小得到区间
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
        #生成0~1的随机数,rand(k,1)代表生成k行1列的随机矩阵，因为是2维的，所以相当于生成k组x，y
    return centroids

def kMeans(dataSet , k ,distMeas = distEclud ,createCent = randCent):
    m = shape(dataSet)[0] # 数据总数
    clusterAssment = mat(zeros((m,2))) # 簇分配结果矩阵，一维代表簇索引值，二维代表误差（当前点到簇质心的距离）
    centroids = createCent(dataSet,k)#随机质心
    clusterChanged = True
    while clusterChanged: # 迭代：计算质心->分配
        clusterChanged = False
        for i in range(m):
            minDist = inf ; minIndex = -1
            for j in range(k): # 遍历所有数据，找到距离每个点最近的质心，即第i个点距离第j个质心最近
                distJI = distMeas(centroids[j,:],dataSet[i,:])# 两点之间的距离公式
                if distJI < minDist:
                    minDist = distJI ; minIndex = j
            if clusterAssment[i,0] != minIndex : clusterChanged = True # 如果任一点簇分配结果发生改变，更新标志
            clusterAssment[i,:] = minIndex,minDist**2
        # print centroids
        for cent in range(k):# 遍历质心更新取值
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0) # axis沿列进行均值计算
    return centroids,clusterAssment

# 二分k-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] # create a list with one centroid
    for j in range(m):# calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]# get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])# compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    import matplotlib.pyplot as plt
    numSamples, dim = dataSet.shape
    if dim != 2:
        print ("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("Sorry! Your k is too large! ")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])  # 为样本指定颜色
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

if __name__ == "__main__":
    datMat = mat(loadDataSet("testSet.txt"))
    # print datMat
    centroids ,clusterAssment= kMeans(datMat,4)
    # centroids ,clusterAssment= biKmeans(datMat,4)
    print clusterAssment

    showCluster(datMat,4, centroids ,clusterAssment)

    import matplotlib.pyplot as plt
    myMat = datMat
    plt.plot(myMat[:, 0], myMat[:, 1], 'ro')
    plt.show()

