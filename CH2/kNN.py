# -*- coding: utf-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, k, labels):
    dataSetSize = dataSet.shape[0]
    #在列方向上重复indX 1次，在行方向重复indX datasize次
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistancesIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voleIlabel = labels[sortedDistancesIndicies[i]]
        classCount[voleIlabel] = classCount.get(voleIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):

    fileObj = open(filename)
    allData = fileObj.readlines()
    lineNumber = len(allData)
    #lineNumber=row, 3=col
    returnMat = zeros((lineNumber, 3))
    classLabelVector = []
    for index, eachLine in enumerate(allData):
        eachLine = eachLine.strip()
        listEachLine = eachLine.split('\t')
        returnMat[index, :] = listEachLine[0:3]
        classLabelVector.append(int(eachLine[-1]))
    fileObj.close()
    return returnMat, classLabelVector

def autoNorm(dataSet):
    #归一化数据
    minVals = dataSet.min(0)#每列的最小值
    maxVals = dataSet.max(0)
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    ranges = maxVals - minVals
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = dataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    #取前10%的数据做测试样本，后面90%为训练样本
    hoRatio = 0.10
    datingMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingMat)
    m = datingMat.shape[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],3, datingLabels[numTestVecs:m])
        #print 'classifyReuslt: {}; datingLabels: {}'.format(classifyResult, datingLabels[i])
        if classifyResult != datingLabels[i]:
            errorCount += 1.0
    print 'the total error rate is : {}'.format(errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['Not at all', 'A little ', 'Very much']
    percentTats = 10
    ffMiles = 10000
    iceCream = 0.5
    datingMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingMat)
    classifyResult = classify0((array([percentTats, ffMiles, iceCream])-minVals)/ranges,normMat, 3, datingLabels)
    print resultList[classifyResult]

def Main1():
    # group, labels = createDataSet()
    # print classify0([1,1],group,3,labels)
    datingMat, datingLabels = file2matrix('datingTestSet2.txt')
    # print datingLabels
    figure = plt.figure()
    # 111 means: 1x1 sub plots, first one
    ax = figure.add_subplot(111)
    # ax.scatter(datingMat[:,0], datingMat[:,1], c=15.0*array(datingLabels),s=15.0*array(datingLabels))

    type1_x1 = []
    type1_y1 = []
    type2_x2 = []
    type2_y2 = []
    type3_x3 = []
    type3_y3 = []

    for i in range(len(datingMat)):
        if datingLabels[i] == 1:
            type1_x1.append(datingMat[i][0])
            type1_y1.append(datingMat[i][1])
        elif datingLabels[i] == 2:
            type2_x2.append(datingMat[i][0])
            type2_y2.append(datingMat[i][1])
        elif datingLabels[i] == 3:
            type3_x3.append(datingMat[i][0])
            type3_y3.append(datingMat[i][1])
        else:
            print 'ERROR!'
    type1 = ax.scatter(type1_x1, type1_y1, s=20, c='red')
    type2 = ax.scatter(type2_x2, type2_y2, s=30, c='green')
    type3 = ax.scatter(type3_x3, type3_y3, s=40, c='blue')

    plt.ylabel(u'玩游戏所耗时间比')
    # plt.ylabel(u'每周消耗冰淇淋数量')
    plt.xlabel(u'每年赢得的飞行常客里程')
    ax.legend((type1, type2, type3), (u'不喜欢', u'一般', u'很有魅力'))
    plt.show()

def Main2():
    classifyPerson()

if __name__ == '__main__':

    sys.exit(Main2())


