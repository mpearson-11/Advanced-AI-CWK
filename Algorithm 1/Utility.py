#   @Copywright Max Pearson
#   Student ID: B123103
#   Data Class
#   External Sources include:
#       http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python


from random import *

#Scientific Packages for plotting and array manipulation
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from matplotlib.legend_handler import HandlerLine2D
import sys


def chosenNetworkModelWeights():
    w=setWeights(15,15,0.0) 
    w[0][8] = -4.41480199671
    w[1][8] = 1.10072113624
    w[2][8] = 0.7642198062
    w[3][8] = 1.06109043196
    w[4][8] = 0.635658299047
    w[5][8] = 0.883320025994
    w[6][8] = 0.812257227145
    w[7][8] = -2.32400104114
    w[0][9] = 1.25257784096
    w[1][9] = 0.951462013908
    w[2][9] = 0.789102909712
    w[3][9] = 0.7884227912
    w[4][9] = 1.15676519221
    w[5][9] = 0.860275053496
    w[6][9] = 0.944644627254
    w[7][9] = 0.921194104769
    w[0][10] = 0.274692904998
    w[1][10] = 0.744367853446
    w[2][10] = 1.06684285828
    w[3][10] = 0.987977474516
    w[4][10] = 0.915855775716
    w[5][10] = 0.756832505855
    w[6][10] = 0.93105801632
    w[7][10] = 0.415479464925
    w[0][11] = 0.943857476099
    w[1][11] = 1.09228962643
    w[2][11] = 1.12862510861
    w[3][11] = 0.986423404717
    w[4][11] = 0.831044381837
    w[5][11] = 0.718898975933
    w[6][11] = 0.788605593635
    w[7][11] = 0.794865081235
    w[0][12] = 1.01060375289
    w[1][12] = 0.903865530787
    w[2][12] = 0.831950743054
    w[3][12] = 0.65454568421
    w[4][12] = 0.856240578802
    w[5][12] = 1.00227604606
    w[6][12] = 0.71004962012
    w[7][12] = 0.826300736104
    w[0][13] = 0.139205291935
    w[1][13] = 0.992662460068
    w[2][13] = 0.975055388585
    w[3][13] = 1.10870241307
    w[4][13] = 0.832036702362
    w[5][13] = 0.855762564705
    w[6][13] = 0.795789492468
    w[7][13] = -0.51353329694

    w[8][14]= -4.52979398322
    w[9][14] = 0.446423284147
    w[10][14] = -1.2227476045
    w[11][14] = -0.0748956050026
    w[12][14] = -0.0856776458392
    w[13][14] = -2.72166514612

    return w

def vector(n):
    vector = []
    for i in range(n):
        vector.append(i)
    return vector

def plotNetworks(Network,title,save):
        
    plt.figure()

    
    l1 = Network[0]["array"]
    l2 = Network[1]["array"]
    l3 = Network[2]["array"]
    l4 = Network[3]["array"]
    l5 = Network[4]["array"]
    l6 = Network[5]["array"]

    line1,=plt.plot(l1,label=Network[0]["label"])
    line2,=plt.plot(l2,label=Network[1]["label"])
    line3,=plt.plot(l3,label=Network[2]["label"])
    line4,=plt.plot(l4,label=Network[3]["label"])
    line5,=plt.plot(l5,label=Network[4]["label"])
    line6,=plt.plot(l6,label=Network[5]["label"])

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={line3: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={line4: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={line5: HandlerLine2D(numpoints=4)})
    plt.legend(handler_map={line6: HandlerLine2D(numpoints=4)})

    plt.ylim((0,1))

    plt.xlabel("Epoch Number")

    plt.title(title)

    plt.savefig(save+".pdf")

############################################################
def viaTan(ar,minM,maxM):
    
    e_min=minM
    e_max=maxM

    normalised=[]
    for i in range(len(ar)):

        top=float(ar[i] - e_min)
        bottom=float(e_max - e_min)

        normal = (2 * top) - 1
        normalised.append(normal)

    return normalised

def viaSigma(ar,minM,maxM):
    
    e_min=minM
    e_max=maxM

    normalised=[]
    for i in range(len(ar)):
        normal= float((ar[i] - e_min))/ float(e_max - e_min)
        normalised.append(normal)

    return normalised

def normalise(ar,minM,maxM):
    return viaSigma(ar,minM,maxM)

############################################################

def setWeights(i,j,n):
    vector=[]
    for x in range(i):
        vector.append([n] * j)
    return vector

############################################################
def hyperbolic_tangent(n):
    return math.tanh(n)
############################################################
def hyperbolic_tangent_dv(n):
    return (1.0 - math.pow(n,2) )
############################################################
def generateRandFor(n):
    # Return Random Number ( a <= n < b )
    
    size = 2.0 / n
   
    a = -size
    b = size
    number = (b - a) * random() + a
    return number

############################################################
def sigmoid(n):
    try:
        value = -1.0 * n
        return 1.0 / (1.0 + math.exp(value) )
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
############################################################
def sigmoid_dv(n):
    return ( 1.0 - n ) * n
############################################################
#Activation
def activation_function(n):
    #return hyperbolic_tangent(n)
    return sigmoid(n)
############################################################
def derivative_function(n):
    #return hyperbolic_tangent_dv(n)
    return sigmoid_dv(n)
############################################################

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
    minimum=0
    maximum=0

    maxim=[]
    minim=[]

    for i in range(start,end):
        #Take from each column
        array=[]
        for j in range(nSet):
            inputNum = (data.getBy(str(j))[i])
            array.append(inputNum)

        maxm=max(array)
        minm=min(array)

        maxim.append(maxm)
        minim.append(minm)
    
    minimum=min(minim)
    maximum=max(maxim)  

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
############################################################


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
            
   ############################################################
    def xlsx(self,fname):
        # Obtained From: 
        # http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python
        # xlsx conversion
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
    ############################################################
    def getBy(self,tag):
        returnArray=[]
        for newObject in self.document:
            returnArray.append(newObject[tag])
        return returnArray
