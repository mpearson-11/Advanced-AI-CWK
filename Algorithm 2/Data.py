#   @Copywright Max Pearson
#   Student ID: B123103
#   Data Class
#   External Sources include:
#       http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python

def normalise(ar,minM,maxM):
    
    e_min=minM
    e_max=maxM

    normalised=[]
    for i in range(len(ar)):
        normal=float((ar[i] - e_min)) / float((e_max - e_min))
        normalised.append(normal)

    return normalised

def NOT_AND(a,b):
    return  int(not(bool(a)) and bool(b) )

def X_OR(a,b):
    return int ( bool(a) ^ bool(b) )

def AND_F(a,b):
    return int ( bool(a) and bool(b) )

def O_R(a,b):
    return int ( bool(a) or bool(b) )


TEST_ME=[
    
    [[1,2,3],[6]],
    [[1,1,2],[4]],
    [[1,2,4],[7]],
    [[1,4,5],[10]],
    [[2,3,1],[6]],
    [[1,3,4],[8]],
    [[2,4,5],[11]]
    
]


OR =[
    [[0,0,0],[ O_R(O_R(0,0),0) ]],
    [[0,0,1],[ O_R(O_R(0,0),1) ]],
    [[0,1,0],[ O_R(O_R(0,1),0) ]],
    [[0,1,1],[ O_R(O_R(0,1),1) ]],
    [[1,0,0],[ O_R(O_R(1,0),0) ]],
    [[1,1,0],[ O_R(O_R(1,1),0) ]],
    [[1,1,1],[ O_R(O_R(1,1),1) ]]
]
OR_TEST=[
    [[0,0,0],[ O_R(O_R(0,0),0) ]],
    [[0,0,1],[ O_R(O_R(0,0),1) ]],
    [[0,1,0],[ O_R(O_R(0,1),0) ]],

    [[1,0,0],[ O_R(O_R(1,0),0) ]],
    [[1,1,0],[ O_R(O_R(1,1),0) ]],

]

AND = [
    [[0,0,0],[ AND_F(AND_F(0,0),0) ]],
    [[0,0,1],[ AND_F(AND_F(0,0),1) ]],
    [[0,1,0],[ AND_F(AND_F(0,1),0) ]],
    [[0,1,1],[ AND_F(AND_F(0,1),1) ]],
    [[1,0,0],[ AND_F(AND_F(1,0),0) ]],
    [[1,1,0],[ AND_F(AND_F(1,1),0) ]],
    [[1,1,1],[ AND_F(AND_F(1,1),1) ]]
]
AND_TEST = [
    [[0,0,0],[ AND_F(AND_F(0,0),0) ]],
    [[0,1,0],[ AND_F(AND_F(0,1),0) ]],
    [[1,0,0],[ AND_F(AND_F(1,0),0) ]],
]

XOR = [
    [[0,0,0],[X_OR(X_OR(0,0),0)]],
    [[0,0,1],[X_OR(X_OR(0,0),1)]],
    [[0,1,0],[X_OR(X_OR(0,1),0)]],
    [[0,1,1],[X_OR(X_OR(0,1),1)]],
    [[1,0,0],[X_OR(X_OR(1,0),0)]],
    [[1,1,0],[X_OR(X_OR(1,1),0)]],
    [[1,1,1],[X_OR(X_OR(1,1),1)]],
]

XOR_TEST = [
    [[0,0,0],[X_OR(X_OR(0,0),0)]],
    [[0,1,0],[X_OR(X_OR(0,1),0)]],
    [[1,0,0],[X_OR(X_OR(1,0),0)]],
]
NOT_AND = [
    [[0,0,0],[0]],
    [[0,0,1],[1]],
    [[0,1,0],[1]],
    [[0,1,1],[0]],
    [[1,0,0],[1]],
    [[1,1,0],[0]],
    [[1,1,1],[1]]
]
NOT_AND_TEST = [
    [[0,0,0],[0]],
    [[0,1,0],[1]],
    [[1,0,0],[1]],
]

import os
import math
#from pylab import *
#import numpy as np

###########################################################################
#   Return configured data to use in Neural Network with 
#   Training and Testing.
###########################################################################
def createNormalisedDataSet(start,end,data,nSet):
    #----------------------------------------------------------------------
    # Create Empty Array
    pat=[]
    minimum=data.minN
    maximum=data.maxN
    #----------------------------------------------------------------------
    #Take Data from Document starting at (start) and ending at (end)
    for i in range(start,end):
        
        innerArray1,innerArray2 = ([],[])
        #----------------------------------------------------------------------
        #Take from each column
        for j in range(nSet):
            inputNum = (data.getBy(str(j))[i])
            innerArray1.append(inputNum)

        #----------------------------------------------------------------------
        #Output (Actual to Compare to Prediction)
        outputNum = data.getBy("8")[i]

         #Normalise Inner Array
        innerArray1=normalise(innerArray1,minimum,maximum)
        
        #Populate Inner arrays
        innerArray2.append(outputNum)
        innerArray2=normalise(innerArray2,minimum,maximum)

        pat.append( [innerArray1,innerArray2] )

       
    #----------------------------------------------------------------------

    return pat

###########################################################################
#   Return configured data to use in Neural Network with 
#   Training and Testing.
###########################################################################
def createDataSet(start,end,data):
    #----------------------------------------------------------------------
    # Create Empty Array
    pat=[]
    #----------------------------------------------------------------------
    #Take Data from Document starting at (start) and ending at (end)
    for i in range(start,end):
        
        innerArray1,innerArray2,innerArray3 = ([],[],[])
        #----------------------------------------------------------------------
        #Take from each column
        for j in range(8):
            inputNum = (data.getBy(str(j))[i])
            innerArray1.append(inputNum)

       
        #----------------------------------------------------------------------
        #Output (Actual to Compare to Prediction)
        outputNum = data.getBy("8")[i]

         #Normalise Inner Array
        innerArray1=innerArray1
        
        #Populate Inner arrays
        innerArray2.append(outputNum)
        innerArray2=innerArray2

        innerArray3.append(innerArray1)
        innerArray3.append(innerArray2)

        pat.append(innerArray3)
    #----------------------------------------------------------------------

    return pat

def showLegend(Data):
    os.system('clear')
    for i,j in Data.tagLine.items():
        print "{0} = {1}".format(i,j)

class Data:
    def __init__(self):
        self.document=[]
        self.tagLine={}
        self.populate()
        self.maxN=self.getMax()
        self.minN=self.getMin()

    def getMax(self):
        maximums=[]
        for i in range(9):
            currentMax=max(self.getBy(str(i)))
            maximums.append(currentMax)

        return max(maximums)

    def getMin(self):
        mimimums=[]
        for i in range(9):
            currentMin=min(self.getBy(str(i)))
            mimimums.append(currentMin)

        return min(mimimums)

    def populate(self):
        doc=self.xlsx('CWData.xlsx')


        for i in range(1,10):
            self.tagLine[str(i-1)]=doc[i]["M"]
            
        for i in range(1,len(doc)):
            newObject={}
            newObject["0"]=float(doc[i]["A"])
            newObject["1"]=float(doc[i]["B"])
            newObject["2"]=float(doc[i]["C"])
            newObject["3"]=float(doc[i]["D"])
            newObject["4"]=float(doc[i]["E"])
            newObject["5"]=float(doc[i]["F"])
            newObject["6"]=float(doc[i]["G"])
            newObject["7"]=float(doc[i]["H"])
            newObject["8"]=float(doc[i]["I"])
            self.document.append(newObject)
            

    def xlsx(self,fname):
        #http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python
        #xlsx conversion
        import zipfile
        from xml.etree.ElementTree import iterparse
        z = zipfile.ZipFile(fname)
        strings = [el.text for e, el in iterparse(z.open('xl/sharedStrings.xml')) if el.tag.endswith('}t')]
        rows = []
        row = {}
        value = ''
        for e, el in iterparse(z.open('xl/worksheets/sheet1.xml')):
            if el.tag.endswith('}v'):
                value = el.text
            if el.tag.endswith('}c'): 
                if el.attrib.get('t') == 's':
                    value = strings[int(value)]
                letter = el.attrib['r']
                while letter[-1].isdigit():
                    letter = letter[:-1]
                row[letter] = value
                value = ''
            if el.tag.endswith('}row'):
                rows.append(row)
                row = {}
        return rows

    def getBy(self,tag):
        returnArray=[]
        for newObject in self.document:
            returnArray.append(newObject[tag])
        return returnArray
