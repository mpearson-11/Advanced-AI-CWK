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

def vector(n):
    vector = []
    for i in range(n):
        vector.append(i)
    return vector

def smooth_plot(actual,pred,name):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    x1 =np.array( vector( len(actual) ) )
    y1=np.array(actual)

    x2 =np.array( vector( len(pred) ) )
    y2=np.array(pred)

    
    x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
    y_smooth1 = spline(x1, y1, x_smooth1)
    
    
    x_smooth2 = np.linspace(x2.min(), x2.max(), 200)
    y_smooth2 = spline(x2, y2, x_smooth2)

    ax.plot(x_smooth1,y_smooth1,'r')
    ax.plot(x_smooth2,y_smooth2,'b')


    plt.title("output (red), prediction (blue)")

    fig.savefig(name+".pdf")
    

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

############################################################
def setWeights(i,j):
    vector=[]
    for x in range(i):
        vector.append([0.0] * j)
    return vector
############################################################
############################################################
def hyperbolic_tangent(n):
    return math.tanh(n)
############################################################
def hyperbolic_tangent_dv(n):
    return (1.0 - math.pow(n,2))
############################################################
def generateRandFor(size):
    # Return Random Number ( a <= n < b )
    a = -2.0 / size
    b = 2.0 / size
    number = (b - a) * random() + a
    return number

############################################################
def sigmoid(n):
    value = -1.0 * n
    return 1.0 / (1.0 + math.exp(value))
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
def anneal(x,r):
    bottom = 1.0 + math.exp(10.0 - (20.0 * x / r) ) 
    return 1.0 - (1.0 / bottom)
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
