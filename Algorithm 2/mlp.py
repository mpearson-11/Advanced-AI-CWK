#   @Copywright Max Pearson
#   Student ID: B123103
#   Date Created 19/03/2015
#
#   Multilayer Perceptron using Backpropagation algorithm with Neuron states
#   
#   External Sources include:
#       http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python

#--------------------------------------------------------------------------
#Standard Libraries for calculations
import random
import math
import sys
#--------------------------------------------------------------------------
#Data Class for Neural Network use
from Utility import *
#--------------------------------------------------------------------------
#Scientific Packages for plotting and array manipulation
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from NetworkManager import *
from cPickle import *

#===============================================================
#===============================================================
class Neuron:

    def __init__(self,i):
        
        self.activation=1.0
        self.n=i
        self.delta=0.0
        self.S=0.0
        self.bias=1.0

    def setDelta(self,delta):
        if delta > 1.0:
            self.delta = 1.0 / delta
        else:
            self.delta = delta

    def activate(self):
        try:
            self.activation=activation_function( self.S )
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    def setSum(self,sumV,LR):
        self.S = sumV + self.bias
        self.activate()
#===============================================================
#===============================================================
class Layer:
    def __init__(self,name,startIndex,length):
        self.name=name
        self.Neurons=[]
        self.__createLayer(startIndex,length)
        print "Neurons Created"
    
    def __createLayer(self,start,length):
        for i in range(start,(start+length)):
            neuron=Neuron(i)
            self.Neurons.append(neuron)
#===============================================================
#===============================================================
class Layers:
    
    def __init__(self):
        
        self.layers=[]
        self.NumNeurons=0
        self.NumLayers=0
        
    def addLayer(self,name,length):
        
        layer=Layer(name,self.NumNeurons,length)
        self.layers.append(layer)

        self.__updateWeightSpace()
        print "Added Layer : "+str(name)
        self.updateLayerSize()

    def updateLayerSize(self):
        countL=0
        countN=0
        for i in self.layers:
            for j in i.Neurons:
                countN+=1
            countL+=1
        self.NumNeurons=countN
        self.NumLayers=countL

    def __updateWeightSpace(self):
        self.weights=setWeights(self.NumNeurons+1,self.NumNeurons+1,0.0)
        
        self.changes=setWeights(self.NumNeurons+1,self.NumNeurons+1,1.0)

#===============================================================
#===============================================================
class NN:
    def __init__(self,inputs,hidden,outputs,name):
        
        self.netName=name
        self.fileName=""

        self.states=[]
        self.learning_rate=0.5
        self.momentum=0.1

        #Add Input,Hidden and Output Layers
        self.Network=Layers()
        self.Network.addLayer("Input",inputs)
        self.Network.addLayer("Hidden",hidden)
        self.Network.addLayer("Output",outputs)
        self.initialiseWeights()
#===============================================================
    def createFile(self,name):
        self.fileName=self.netName+"_"+name
        File=open(self.fileName,"w")
        File.close()
#===============================================================
    def saveOutput(self,output):
        File=open(self.fileName,"a")
        File.write(output)
        File.close()
#===============================================================
    def getWeights(self,i1,i2):
        return self.Network.weights[i1][i2]
#===============================================================
    def getChanges(self,i1,i2):
        return self.Network.changes[i1][i2]
#===============================================================
    def getLayer(self,index):
        return self.Network.layers[index]
#===============================================================   
    def getLayerNeurons(self,index):
        return self.getLayer(index).Neurons
#===============================================================
    def getLayersNeuron(self,layer,index):
        return self.getLayerNeurons(layer)[index]
#===============================================================
    def getPrediction(self):
        return self.getLayersNeuron(2,0).activation
#===============================================================
    def feed_forward(self,inputs):
        
        output=""
        output+="\n--------------------------------------------------------------"
        output+="\n\tFeed Forward"
        output+="\n--------------------------------------------------------------\n"
        

        for i in range(len(self.getLayerNeurons(0))):
            self.getLayerNeurons(0)[i].activation=inputs[i]
            output+="\na[i]: {0}".format(self.getLayerNeurons(0)[i].activation)
        output+="\n-------------------------------------------\n"
        

        for nodeJ in self.getLayerNeurons(1):
            sumJ=self.feed_layer(0,nodeJ)
            nodeJ.setSum(sumJ,self.learning_rate)
            output+="\na[j]= {0} , S[j]= {1}".format(nodeJ.activation,nodeJ.S)
        output+="\n-------------------------------------------\n"
        
        for nodeK in self.getLayerNeurons(2):
            sumK=self.feed_layer(1,nodeK)
            nodeK.setSum(sumK,self.learning_rate)
            output+="\na[k]= {0} , S[k]= {1}".format(nodeK.activation,nodeK.S)
        

        output+="\n--------------------------------------------------------------"
        output+="\n--------------------------------------------------------------\n\n"
        self.saveOutput(output)
#===============================================================
    def feed_layer(self,layer,nodeForward):
        sumOf=0.0
        for nodeBack in self.getLayerNeurons(layer):
            sumOf += (self.getWeights(nodeBack.n,nodeForward.n) * nodeBack.activation)
        return sumOf
#===============================================================
    def backPropagation(self,targets,epoch):
        #Output Cells
        output=""
        output+="\n--------------------------------------------------------------"
        output+="\n\tBackpropagate Epoch: {0}".format(epoch)
        output+="\n--------------------------------------------------------------"
        for nodeK in self.getLayerNeurons(2):
            error = targets[0] - nodeK.activation
            delta = derivative_function( nodeK.activation ) * error
            nodeK.setDelta(delta)
            output+="\n\nFind Delta k[{0}]".format(nodeK.n)
            output+="\nz[k]={0} \ntarget[k]={1} \na[k] ={2} \ndelta=dv(z[k])* ( target[k] - a[k] )\n".format(nodeK.S,targets[0],nodeK.activation)
            output+="\ndelta = dv({0}) * ({1} - {2})".format(nodeK.S,targets[0],nodeK.activation)
            output+="\n= {0}".format(nodeK.delta)

        #Hidden Cells
        output+="\n------------------------------------\n"
        for nodeJ in self.getLayerNeurons(1):
            error=0.0
            for nodeK in self.getLayerNeurons(2):
                error+= ( nodeK.delta * self.getWeights(nodeJ.n,nodeK.n) )

            delta = derivative_function( nodeJ.activation  ) * error
            nodeJ.setDelta(delta)
            output+="\n\nFind Delta j[{0}]".format(nodeJ.n)
            output+="\nz[j]={0} \nerror = sigma( delta[k] * w[j][k] ) = {1} \ndelta = dv(z[j]) * error \n".format(nodeJ.S,error)
            output+="\ndelta = dv({0}) * {1}".format(nodeJ.S,error)
            output+="\n= {0}".format(nodeJ.delta)

        output+="\n------------------------------------\n"
        output+="\n--------------------------------------------------------------"
        output+="\n--------------------------------------------------------------\n\n"
        self.saveOutput(output)
 #===============================================================       
    def updateWeights(self,epoch):
        output="\n--------------------------------------------------------------"
        output+="\n\tUpdate Weights"
        output+="\n--------------------------------------------------------------\n"

        self.saveOutput(output)

        for nodeK in self.getLayerNeurons(2):
            self.updateLayerBack(1,nodeK)

        for nodeJ in self.getLayerNeurons(1):
            self.updateLayerBack(0,nodeJ)

#===============================================================
    def updateLayerBack(self,layer,nodeForward):
        output=""
        for nodeBack in self.getLayerNeurons(layer):
            
            change = nodeBack.activation * nodeForward.delta

            self.Network.weights[nodeBack.n][nodeForward.n] = \
            self.getWeights(nodeBack.n,nodeForward.n) + \
            (self.learning_rate * change) + \
            (self.momentum      * self.getChanges(nodeBack.n,nodeForward.n))
            output+="\n--------------------------------------"
            output+="\nWeight: {0},{1} = \nw({2}) + \n\t(LR({3}) * CH({4}) ) + \n\t\t( M({5}) * CH*({6}) ) \n".format(nodeBack.n,nodeForward.n,\
                self.getWeights(nodeBack.n,nodeForward.n),self.learning_rate,change,self.momentum,self.getChanges(nodeBack.n,nodeForward.n))

            self.Network.changes[nodeBack.n][nodeForward.n]=change
        
        self.saveOutput(output)
#===============================================================
    def weightDecay(self,n,weight):
        v = 1 / self.learning_rate * n
        sumOf=0.0
        for i in range(len(weight)):
            sumOf+=math.pow(weight[i],2)
        omega=sumOf / 2.0
        return np.array(weight) + ( v * omega ) 
#===============================================================
    def simulated_annealing(self,error,totalEpochs):
        p=0.01
        q=self.learning_rate
        r=totalEpochs
        x=error
        return p + ((q - p) * anneal(x,r) )
#===============================================================
    def train(self,examples,name):
        epoch=0
        exitCounter=0
        self.createFile(name+".txt")
        print "Training on :"+name
        while True:
            Errors=[]
            for values in examples:
                
                inputs  = values[0]
                outputs = values[1]

                self.feed_forward(inputs)
                self.backPropagation(outputs,epoch)
                self.updateWeights(epoch)

                error = self.getError(outputs)
                Errors.append(error)
            
            print "Epoch :{0}".format(epoch)
            #exitCounter=self.bold_driver(Errors,exitCounter)
            epoch+=1

            if exitCounter > 10 or epoch > 500:
                print "Epochs: "+str(epoch)
                print "Exit Counter: "+str(exitCounter)
                break
#===============================================================
    def test(self,examples):
        self.createFile("Test.txt")

        predictions=[]
        actualValues=[]

        for values in examples:
            self.feed_forward(values[0])
            actual=values[1]
            predictor=self.getPrediction()
            print "\nActual : {0} Prediction: {1}".format(actual[0],predictor)
            predictions.append(predictor)
            actualValues.append(actual[0])

        smooth_plot(actualValues,predictions,"ActualVsPredictions","output (red), prediction (blue)")
#===============================================================
    def initialiseWeights(self):

        for nodeJ in self.getLayerNeurons(1):
            self.randomLayerBack(0,nodeJ)

        for nodeK in self.getLayerNeurons(2):
            self.randomLayerBack(1,nodeK)

    def randomLayerBack(self,layer,nodeForward):
        
        size=len(self.getLayerNeurons(layer))
        for nodeBack in self.getLayerNeurons(layer):
            self.Network.weights[nodeBack.n][nodeForward.n]=generateRandFor(layer)

#===============================================================
    def getError(self,actualValues):
        sumOf=0.0
        error=0.0
        for neuron in self.getLayerNeurons(2):
            sumOf+= math.pow( (actualValues[0] - neuron.activation) , 2)
        error = math.sqrt( sumOf / 2.0)
        return error
#===============================================================
    def bold_driver(self,ERRORS,valNum):
        size=len(ERRORS)
        inc=1.1
        dec=0.5
        if size > 1:
            new=ERRORS[size-1]
            old=ERRORS[size-2]
            if (new < old):
                if (self.learning_rate * inc)  >= 0.9:
                    self.learning_rate = 0.9
                else:
                    self.learning_rate *= inc
                valNum=0
            elif new > old:
                self.learning_rate *= dec
                valNum+=1
            else:
                pass
        return valNum
#===============================================================
#===============================================================

data = Data()
#----------------------------------------------------------------------
# Create Training and Test Data from CWKData.xlsx
# Take 80% of data (400 epochs as default) and leave rest for testing

DATA_SET        =   200
INPUTS          =   4

#Data Variables
T_START         =   2
T_END           =   T_START    + int(DATA_SET * 0.7)

TTS_START       =   T_END
TTS_END         =   TTS_START  + int(DATA_SET * 0.15)

TST_START       =   TTS_END
TST_END         =   TST_START  + int(DATA_SET * 0.15)    


#CREATE NORMALISED DATA FROM FILE
TRAINING_DATA   = createNormalisedDataSet(  T_START,   T_END,   data,  INPUTS)
VALIDATION_DATA = createNormalisedDataSet(  TTS_START, TTS_END, data,  INPUTS)
TESTING_DATA    = createNormalisedDataSet(  TST_START, TST_END, data,  INPUTS)

inputs          = len(TRAINING_DATA[0][0])
hidden = 2
outputs         = len(TRAINING_DATA[0][1])

Network = NN(inputs,hidden,outputs,"Network1")

Network.train(TRAINING_DATA,"Testing_Data")

Network.train(VALIDATION_DATA,"Validation_Data")

Network.test(TESTING_DATA)




############################################################
############################################################
############################################################

