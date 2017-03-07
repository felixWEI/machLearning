# -*- coding: utf-8 -*-
from math import log
import operator
def calcShannonEny(dataSet):
    numEntries = len(dataSet)
    labelsCounts = {}
    for dataVec in dataSet:
        currentLabel = dataVec[-1]
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1
    shannonEnt = 0.0
    for tmpKey in labelsCounts.keys():
        prob = float(labelsCounts[tmpKey]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # dataSet = [['s', 's', 'no', 'no'],
    #            ['s', 'l', 'yes', 'yes'],
    #            ['l', 'm', 'yes', 'yes'],
    #            ['m', 'm', 'yes', 'yes'],
    #            ['l', 'm', 'yes', 'yes'],
    #            ['m', 'l', 'no', 'yes'],
    #            ['m', 's', 'no', 'no'],
    #            ['l', 'm', 'no', 'yes'],
    #            ['m', 's', 'no', 'yes'],
    #            ['s', 's', 'yes', 'no']]
    # labels = ['L', 'F', 'H']
    return dataSet, labels
def splitDataSet(dataSet, axes, value):
    #剔除dataSet样本向量第axes个值为value的项，然后返回剩余的
    retDataSet = []
    for feaVec in dataSet:
        if feaVec[axes] == value:
            tempFeaVec = feaVec[:axes]
            tempFeaVec.extend(feaVec[axes+1:])
            retDataSet.append(tempFeaVec)
    return retDataSet

'''
如同例子当中的数据集[1,1,'yes'],我们需要从第一个特征和第二个特征中取出一个用于划分数据集，
这样就需要算出在取出各特征值后的信息增益，即信息熵的增加哪个更大就用哪个特征来划分
'''
def chooseBestFeatureToSplit(dataSet):

    bestInfoGain = 0.0
    bestFeature = -1
    baseShannonEnt = calcShannonEny(dataSet)
    numberFeature = len(dataSet[0]) - 1
    for i in range(numberFeature):
        #把不同的特征值取出
        featureList = [temp[i] for temp in dataSet]
        uniqueVals = set(featureList)

        newShannonEnt = 0.0
        for featureVal in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, featureVal)
            #此处prob是指在dataSet的范围内，得到subDataSet集合的概率。然后计算此subDataSet子集的信息熵
            #子集是指在去掉对应featureVal之后得到的集合
            prob = len(subDataSet) / float(len(dataSet))
            newShannonEnt += prob * calcShannonEny(subDataSet)

        newEntGain = baseShannonEnt - newShannonEnt
        if newEntGain > bestInfoGain:
            baseShannonEnt = newShannonEnt
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [tmp[-1] for tmp in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featVals = [tmp[bestFeature] for tmp in dataSet]
    uniqueVals = set(featVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    firstIndex = featLabels.index(firstStr)
    for tmpKey in secondDict.keys():
        if testVec[firstIndex] == tmpKey:
            if type(secondDict[tmpKey]).__name__ == 'dict':
                classLabel = classify(secondDict[tmpKey], featLabels, testVec)
            else:
                classLabel = secondDict[tmpKey]
    return classLabel

def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(fileName):
    import pickle
    fr = open(fileName,'r')
    return pickle.load(fr)

myDat,myLabel = createDataSet()
#print chooseBestFeatureToSplit(myDat)
myTree = createTree(myDat, myLabel)
print myTree
myLabel = ['no surfacing', 'flippers']
print classify(myTree, myLabel, [1,1])
#