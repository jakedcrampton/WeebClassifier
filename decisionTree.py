import pandas as pd
import math

class treeNode:
    def __init__(self):
        self.isPredictor=False
        self.left=None
        self.right=None
        self.decidingColumn=""
        self.cutoff=0
        self.p=0

    def choosePath(self,row: pd.core.series.Series):
        if (row[self.decidingColumn]<=self.cutoff):
            return self.left
        return self.right

    def predict(self, data:pd.DataFrame):
        self.p=data["Male"].mean() #Yeet for normalized data

class leafNode:
    def __init__(self):
        self.isPredictor = True
        self.p=0
        self.num_examples=0
    def predict(self, data:pd.DataFrame):
        self.p=data["Male"].mean() #Yeet for normalized data
        self.num_examples=len(data)

class decisionTree:
    def __init__(self, treeDepth):
        self.root=None
        self.data=None
        self.treeDepth=treeDepth #Max depth, 1 indexed, including predictor nodes
        self.sizeCutoff=30 #stop when less than this
        self.debug=False
        self.countNodes=0
        self.quantiles=None

    def fit(self, data: pd.DataFrame):
        self.data=data
        self.root=self.makeTree(data, data.iloc[:,4:].columns,1)

    def predict(self,row: pd.core.series.Series):
        node=self.root
        while(not node.isPredictor):
            #This note is a treeNode
            node=node.choosePath(row)
        return node.p
    def isPerfectSplit(self, data: pd.DataFrame):
        return len(data[data["Male"] == 1]) == len(data) or len(data[data["Male"] == 1]) == 0

    def makeTree(self, data: pd.DataFrame, attributes, cur_depth):
        # Model:
        # where Q_q represents the q quantile, which we tune during training
        #     data[x] <= Q_q[x]?
        #      yes/       \no
        #   node.left   node.right
        #

        #Debug Statements
        self.countNodes+=1
        if(self.debug==True):
            print("Num Nodes: " + str(int(self.countNodes))+"/"+str(int(2**self.treeDepth-1)))

        #Finish this branch
        if cur_depth==self.treeDepth or len(data)<self.sizeCutoff or self.isPerfectSplit(data):
            # make a prediction Node
            node=leafNode()
            node.predict(data)
            return node

        #Split in two
        else:
            node=treeNode()
            #Do checks for which thing is best
            jesusObject=self.findBestSplit(data,attributes)#Of form [[data1,data2], column, cutoff]
            rightData=jesusObject[0][1]
            leftData=jesusObject[0][0]
            newAttributes=attributes.drop(jesusObject[1])
            node.decidingColumn=jesusObject[1]
            node.cutoff=jesusObject[2]
            node.predict(data)
            node.left= self.makeTree(leftData,newAttributes,cur_depth+1)
            node.right= self.makeTree(rightData,newAttributes,cur_depth+1)
            return node

    def findBestSplit(self,data:pd.DataFrame, attributes):
        best_quantile=0
        min_score=2
        answer=[] #Of form [[data1,data2], column, cutoff]
        quantiles=[0.17, .33, 0.5, .67, .83]
        if(not self.quantiles is None):
            quantiles=self.quantiles
        for attribute in attributes:
            for quantile in quantiles:
                val = data[attribute].quantile(q=quantile, interpolation='midpoint')
                splitData=[data[data[attribute]<=val],data[data[attribute]>val]]
                score=2
                if self.isPerfectSplit(splitData[0]) or self.isPerfectSplit(splitData[1]):
                    score=0
                else:
                    score=self.Remainder(splitData)
                if(score<min_score):
                    min_score=score
                    answer=[splitData,attribute,val]
                    best_quantile=quantile
        if(self.debug==True):
            print("Quantile Chosen: "+str(best_quantile))
        return answer
    ###################################################
    # Methods for choosing
    ###################################################
    def Gain(self,data,splitData):#Unused. All useful info can be gotten from remainder
        n=len(data)
        a=len(data[data["Male"]==1])
        b=n-a
        return self.I(a/n,b/n)-self.Remainder(splitData)
    def I(self,a,b):
        if(a==0 or b==0):
            return 0
        return -1*a*math.log(a,2)-1*b*math.log(b,2)
    def Remainder(self,splitData):
        data1=splitData[0]
        data2=splitData[1]
        n=len(data1)+len(data2)
        n1=len(data1)
        a1=len(data1[data1["Male"]==1])
        b1=n1-a1
        n2 = len(data2)
        a2 = len(data2[data2["Male"] == 1])
        b2 = n2 - a2
        return n1*self.I(a1/n1,b1/n1)/n+n2*self.I(a2/n2,b2/n2)/n
