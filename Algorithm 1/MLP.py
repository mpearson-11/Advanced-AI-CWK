#   Student Name: Max Pearson
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
from matplotlib.legend_handler import HandlerLine2D
from cPickle import *


#Plot Network
def plotNN(Network):
    Network.plotErrors()
    Network.plotLearning()
    Network.plotPrediction()

#Save Network as Objet for Later use
def saveNN(Network,name):
    File=open(name+"Object.bin","w")
    dump(Network,File)
    File.close()

#Load Object
def loadNN(name):
    File=open(name+"Object.bin","r")
    Network=load(File)
    File.close()
    return Network


#Load Network Chosen State
def loadNetwork(networkName):
    File=open(networkName,"r")
    OBJECT = load( File )
    File.close()
    return OBJECT

#Show State
def showNetwork(network):
    NETWORK = loadNetwork(network+".txt")

    learning=NETWORK["learning"]
    
    changes=NETWORK["changes"]
    weights=NETWORK["weights"]

    neuronsI=NETWORK["i"]
    neuronsJ=NETWORK["j"]
    neuronsK=NETWORK["k"]


    print "\n--------------------------------------------------------------"
    print "Input Layer\n"
    for nodeI in neuronsI:
        print "Neuron number : {0} ".format(nodeI.n)
    print "\n--------------------------------------------------------------"
    print "Hidden Layer\n"
    for nodeJ in neuronsJ:
        print "Neuron number : {0} ".format(nodeJ.n)
    print "\n--------------------------------------------------------------"
    print "Output Layer\n"
    for nodeK in neuronsK:
        print "Neuron number : {0} ".format(nodeK.n)
    print "\n--------------------------------------------------------------"
    

    print "\nInput -> Hidden Weights\n"
    for nodeJ in neuronsJ:
        j=nodeJ.n
        
        for nodeI in neuronsI:
            i=nodeI.n 
            print "w ({0} -> {1}) = {2}".format(i,j,weights[i][j]) 
           
    print "\n--------------------------------------------------------------"
    print "\nHidden -> Output Weights\n"
    for nodeK in neuronsK:
        k=nodeK.n
        
        for nodeJ in neuronsJ:
            j=nodeJ.n 
    
            print "w ({0} -> {1}) = {2}".format(j,k,weights[j][k]) 



#===============================================================
#===============================================================
#   Neuron
class Neuron:

    def __init__(self,i):
        
        self.activation=0.0
        self.n=i
        self.delta=0.0
        self.S=0.0
        self.bias=1.0

    def setDelta(self,delta):
        #if delta > 1.0:
            #self.delta = 1.0 / delta
        #else:
        self.delta = delta

    def activate(self):
        try:
            self.activation=activation_function( self.S )
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    def setSum(self,sumV,LR):
        self.S = 1 + sumV
        self.activate()
#===============================================================
#===============================================================
#   Layer
class Layer:
    def __init__(self,name,startIndex,length):
        self.name=name
        self.Neurons=[]
        self.__createLayer(startIndex,length)
    
    def __createLayer(self,start,length):
        for i in range(start,(start+length)):
            neuron=Neuron(i)
            self.Neurons.append(neuron)
#===============================================================
#===============================================================
#   Layers
class Layers:
    
    def __init__(self,chosen):
        
        self.layers=[]
        self.NumNeurons=0
        self.NumLayers=0
        self.chosen=chosen
        
    def addLayer(self,name,length):
        
        layer=Layer(name,self.NumNeurons,length)
        self.layers.append(layer)

        self.__updateWeightSpace()
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
        if self.chosen == 1:
            self.weights = chosenNetworkModelWeights()
        else:
            self.weights=setWeights(self.NumNeurons+1,self.NumNeurons+1,0.0)
        
        self.changes=setWeights(self.NumNeurons+1,self.NumNeurons+1,1.0)

#===============================================================
#===============================================================
#   Main Neural Network
class NN:
    def __init__(self,inputs,hidden,outputs,name):
        
        self.netName=name
        self.fileName=""

        self.states={}
        self.learning_rate=0.5
        self.momentum=0.9
        self.weightDecayFactor=0.03

        #Add Input,Hidden and Output Layers
        if name =="FinalNetwork":
            self.Network=Layers(1)
        else:
            self.Network=Layers(0)

        self.Network.addLayer("Input",inputs)
        self.Network.addLayer("Hidden",hidden)
        self.Network.addLayer("Output",outputs)

        self.initialiseWeights()

        #Training and Validation arrays
        self.trainingErrors=[]
        self.validationErrors=[]
        
        #Learning Rate Array from Training and Validation
        self.learningRates=[]

        #Prediction and Output Arrays based on Testing
        self.testDataPredictions=[]
        self.testDataOutputValues=[]
        self.testErrors=[]

        self.exitCounter=0
        print name +"  Created"
#===============================================================    
    def captureState(self,epoch):
        self.states[str(epoch)]={}
        
        #Error State
        self.states[str(epoch)]["training"]=self.trainingErrors
        self.states[str(epoch)]["validation"]=self.validationErrors
        
        #Weight State
        self.states[str(epoch)]["weights"]=self.getW()
        self.states[str(epoch)]["changes"]=self.getC()
        
        #Learning State
        self.states[str(epoch)]["learning"]=self.learningRates

        #Neurons
        self.states[str(epoch)]["i"]=self.getLayerNeurons(0)
        self.states[str(epoch)]["j"]=self.getLayerNeurons(1)
        self.states[str(epoch)]["k"]=self.getLayerNeurons(2)

        #Only Keep 12 states (delete all others) need 11th state
        if epoch >= 12:
            del self.states[str(epoch-12)]
 #===============================================================       
    def getStateTypeAt(self,epoch,name):
        return self.states[str(epoch)][name]
#===============================================================
    def saveNetwork(self,epoch):
        File=open(self.netName+".txt","w")
        dump(self.states[str(epoch)],File)
        File.close()
        del self.states
#===============================================================
    def getWeights(self,i1,i2):
        return self.getW()[i1][i2]
#===============================================================      
    def getW(self):
        return self.Network.weights
#===============================================================
    def getChanges(self,i1,i2):
        return self.Network.changes[i1][i2]
#===============================================================
    def getC(self):
        return self.Network.changes
#===============================================================
    def getLayer(self,index):
        return self.Network.layers[index]
#===============================================================   
    def getLayerNeurons(self,index):
        return self.getLayer(index).Neurons
#=============================================================== 
    def getLayerSize(self,index):
        return len(self.getLayerNeurons(index))
#===============================================================
    def getLayersNeuron(self,layer,index):
        return self.getLayerNeurons(layer)[index]
#===============================================================
    def getPrediction(self):
        return self.getLayersNeuron(2,0).activation
#===============================================================
    def feed_forward(self,inputs):
        
        #Feed Input Layer
        for i in range(len(self.getLayerNeurons(0))):
            nodeI=self.getLayerNeurons(0)[i]
            nodeI.activation=inputs[i]
          
        #Feed Hidden Layer
        for nodeJ in self.getLayerNeurons(1):
            sumI=0.0
            for nodeI in self.getLayerNeurons(0):
                sumI += (self.getWeights(nodeI.n,nodeJ.n) * nodeI.activation)  
            nodeJ.setSum(sumI,self.learning_rate)

        # Feed Output Layer
        for nodeK in self.getLayerNeurons(2):
            sumJ=0.0
            
            for nodeJ in self.getLayerNeurons(1):
                sumJ += (self.getWeights(nodeJ.n,nodeK.n) * nodeJ.activation)
            nodeK.setSum(sumJ,self.learning_rate)

#===============================================================
    def backPropagation(self,targets):
        
   
        #BackPropagate Output Layer Calculating Changes
        for nodeK in self.getLayerNeurons(2):
            error = targets[0] - nodeK.activation
            delta = derivative_function( nodeK.activation ) * error
            nodeK.setDelta(delta)
            
        #Update Output Weights
        for nodeJ in self.getLayerNeurons(1):
            for nodeK in self.getLayerNeurons(2):
                j=nodeJ.n
                k=nodeK.n
                
                change = nodeK.delta * nodeJ.activation
                oldChange = self.getChanges(j,k)
                
                oldWeight=self.getWeights(j,k)
                
                self.getW()[j][k]= oldWeight+ \
                (self.learning_rate * change) +\
                (self.momentum * oldChange)

                changes=self.getC()
                self.getC()[j][k]=change
        
        #BackPropagate Hidden Layer Calculating Changes
        for nodeJ in self.getLayerNeurons(1):
            error=0.0
            for nodeK in self.getLayerNeurons(2):
                error+= ( nodeK.delta * self.getWeights(nodeJ.n,nodeK.n) )

            delta = derivative_function( nodeJ.activation  ) * error
            nodeJ.setDelta(delta)
            
        #Update Hidden Weights
        for nodeJ in self.getLayerNeurons(1):
            for nodeI in self.getLayerNeurons(0):
                i=nodeI.n
                j=nodeJ.n 
                
                change = nodeJ.delta * nodeI.activation
                oldChange = self.getChanges(i,j)
                
                oldWeight=self.getWeights(i,j)
                
                self.getW()[i][j]= oldWeight+ \
                (self.learning_rate * change) +\
                (self.momentum * oldChange)

                changes=self.getC()
                self.getC()[i][j]=change

#===============================================================
    def decayWeights(self,epoch):
        #Decay weights from hidden to output
        for nodeK in self.getLayerNeurons(2):
            for nodeJ in self.getLayerNeurons(1):
                k=nodeK.n 
                j=nodeJ.n 
                #Decay based on weight changes in last epoch
                self.getC()[j][k] -= (self.weightDecayFactor *  self.getStateTypeAt(epoch-1,"changes")[j][k] )

        #Decay weights from input to hidden
        for nodeJ in self.getLayerNeurons(1):
            for nodeI in self.getLayerNeurons(0):
                j=nodeJ.n 
                i=nodeI.n 
                #Decay based on weight changes in last epoch
                self.getC()[i][j] -= (self.weightDecayFactor *  self.getStateTypeAt(epoch-1,"changes")[i][j] ) 
#===============================================================
    def train(self,examples,iterations):
        #Train Network on given iterations before validation
        outError=0.0
        for i in range(iterations):
            error=0.0
            for values in examples:
                
                inputs  = values[0]
                outputs = values[1]

                self.feed_forward(inputs)
                self.backPropagation(outputs)

                error += self.getError(outputs)

        outError = math.sqrt( error / len(examples) )
        return outError 

#===============================================================     
    def runProgram(self,TRAIN,VALID,n):
        
        print "\nRunning : "+self.netName+"\n"
        oldError=0.0
        error=0.0
        epoch=0
        self.validationErrors=[]
        self.trainingErrors=[]
        self.learningRates=[]

        while True:

            #Train with n iterations
            trainError = self.train(TRAIN,n)
           
            oldError=error
            error=0.0

            for values in VALID:
                
                inputs  = values[0]
                outputs = values[1]

                self.feed_forward(inputs)
                self.backPropagation(outputs)

                error += self.getError(outputs)

            error = math.sqrt( error / len(VALID) )
            
            self.validationErrors.append(error)
            self.trainingErrors.append(trainError)
            self.learningRates.append(self.learning_rate)

            self.bold_driver(error,oldError)
            
            #Capture state to allow weight decay access to weight changes by epoch
            self.captureState(epoch)
            
            if epoch > 1:
                #Decay Weight after first epoch
                self.decayWeights(epoch)
            
            print "Epoch = {0} Error = {1} LR = {2}".format(epoch,error,self.learning_rate)
           
            #checkCount updates termination value
            self.checkCount(error,oldError)

            if self.exitCounter > 8 or epoch > 1000:
                #Go back 10 to minimum epoch/time 

                epochMinima = epoch - 11
                
                #Override all Network Weights for Best test data 
                #based on minima termination weights
                self.validationErrors    = self.getStateTypeAt(epochMinima, 'validation' )
                self.trainingErrors      = self.getStateTypeAt(epochMinima, 'training'   )
                self.Network.weights     = self.getStateTypeAt(epochMinima, 'weights'    )
                self.Network.changes     = self.getStateTypeAt(epochMinima, 'changes'    )

                print "\n\n------------------------------------------"
                print "\nEpochs: "+str(epoch)
                print "\nEpoch Minimum "+str(epochMinima)
                print "\n------------------------------------------"
                print "\n Network {0} saved state under: {1}.txt   ".format(self.netName,self.netName)
                self.saveNetwork(epochMinima)
                break
            
            else:
                epoch+=1

       

#===============================================================
    def test(self,examples):

        self.testDataPredictions=[]
        self.testDataOutputValues=[]
        self.testErrors=[]

        print "\nSPLIT\n\n"
        for values in examples:
            self.feed_forward(values[0])
            actual=values[1]
            predictor=self.getPrediction()
            self.testDataPredictions.append(predictor)
            self.testDataOutputValues.append(actual[0])
            print "Output - {0} Prediction - {1}".format(actual[0],predictor)

            error = self.getError(values[0])
            self.testErrors.append(error)

#===============================================================
    def initialiseWeights(self):

        iSize=self.getLayerSize(0)
        jSize=self.getLayerSize(2)

        #Initialise w[i][j]
        for nodeJ in self.getLayerNeurons(1):
           
            for nodeI in self.getLayerNeurons(0):
                i=nodeI.n 
                j=nodeJ.n 
                self.getW()[i][j]=generateRandFor(iSize)

        #Initialise w[j][k]
        for nodeK in self.getLayerNeurons(2):
           
            for nodeJ in self.getLayerNeurons(1):
                j=nodeJ.n 
                k=nodeK.n 
                self.getW()[j][k]=generateRandFor(jSize)

#===============================================================
    def getError(self,actualValues):
        error=0.0
        for neuron in self.getLayerNeurons(2):
            error+= math.pow( (actualValues[0] - neuron.activation) , 2)
        error = error / len(self.getLayerNeurons(2))
        return error
#===============================================================
    def bold_driver(self,new,old):

        inc=1.1
        dec=0.5
        
        if (new < old):
            if (self.learning_rate * inc)  >= 0.9:
                self.learning_rate = 0.9
            
            else:
                self.learning_rate *= inc
        
        elif new > old:    
            self.learning_rate *= dec
        else:
            pass
#===============================================================
    #   Check Exit Counter
    def checkCount(self,new,old):
        if (new < old):
            self.exitCounter=0

        else:
            self.exitCounter+=1   
#===============================================================
    #   Plot Training and Validation Errors
    def plotErrors(self):

        plt.figure()
        
        x1 =np.array( vector( len(self.trainingErrors) ) )
        y1 = np.array(self.trainingErrors)

        x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
        y_smooth1 = spline(x1, y1, x_smooth1)

        #Validation Plot
        x2 =np.array( vector( len(self.validationErrors) ) )
        y2= np.array(self.validationErrors)

        x_smooth2 = np.linspace(x2.min(), x2.max(), 200)
        y_smooth2 = spline(x2, y2, x_smooth2)

      
        line1,=plt.plot(x_smooth1,y_smooth1,'r',label="training errors")
        line2,=plt.plot(x_smooth2,y_smooth2,'g',label="validation errors")
        
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        
        plt.ylabel("RMS Error")
        plt.xlabel("Epoch Number")

        plt.title(self.netName+" Validation and Training Errors")
        plt.savefig(self.netName+"_Errors.pdf")

#===============================================================
    #   Plot Learning rate changes
    def plotLearning(self):
        plt.figure()
       
        #Learning Rate Plot
        x1 =np.array( vector( len( self.learningRates ) ) )
        y1 = np.array( self.learningRates )

        x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
        y_smooth1 = spline(x1, y1, x_smooth1)

        plt.plot(x_smooth1,y_smooth1,'r')
        plt.ylabel("Learning parameter")

        plt.title(self.netName+" Learning Rates")
        plt.savefig(self.netName+"_Learning.pdf")
#===============================================================
#===============================================================
    #   Plot Predictions
    def plotPrediction(self):
        
        plt.figure()

        x1 = np.array(vector( len(self.testDataOutputValues) ) )
        y1 = np.array(self.testDataOutputValues)

        x2 = np.array(vector( len(self.testDataPredictions) ) )
        y2 = np.array(self.testDataPredictions)

        x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
        y_smooth1 = spline(x1, y1, x_smooth1)
        
        x_smooth2 = np.linspace(x2.min(), x2.max(), 200)
        y_smooth2 = spline(x2, y2, x_smooth2)

        line1,=plt.plot(x_smooth1,y_smooth1,'r',label="output values")
        line2,=plt.plot(x_smooth2,y_smooth2,'b',label="prediction values")

        plt.ylabel("Normalised Data Value")
        plt.xlabel("Epoch Number")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

        plt.title(self.netName+" Predictions")

        plt.savefig(self.netName+"_Predictions.pdf")



def getNetworkArrays(inputs,validation):
    #----------------------------------------------------------------------
    # Create Training and Test Data from CWKData.xlsx
    # Take 70% of data (560 examples as default) and leave rest for testing and vaidation

    data = Data()

    if validation > 0.30:
        print "Data is incorrect"
        exit()


    test = (1.0 - 0.7 - validation)



    DATA_SET        =   560
    INPUTS          =   inputs
    T_START         =   2
    T_END           =   T_START    + int(DATA_SET * 0.7)

    TTS_START       =   T_END
    TTS_END         =   TTS_START  + int(DATA_SET * validation)

    TST_START       =   TTS_END
    TST_END         =   TST_START  + int(DATA_SET * test)  
    
    #CREATE NORMALISED DATA FROM FILE
    train  = createNormalisedDataSet(  T_START,   T_END,   data,  INPUTS)
    valid = createNormalisedDataSet(  TTS_START, TTS_END, data,  INPUTS)
    test   = createNormalisedDataSet(  TST_START, TST_END, data,  INPUTS)

    #Outputs Cannot be changed (data specific)
    outputs         = len(test[0][1])

    return (train,valid,test,outputs)





#Input and Hidden Nodes

def runNetwork():

    data=Data()
    
    NETWORKS_PLOTTING=[]

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 1:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (8 inputs) (8 hidden) ( 1 output >fixed< ) 
    #       10 training iterations per epoch
    #
    #       Learning Rate = 0.5
    #       Momentum      = 0.9
    #       Weight Decay  = 0.03
    #
    (n1_input,n1_hidden) =  (8,8)
    (n_train1,n_valid1,n_test1,n_output1) = getNetworkArrays(n1_input,0.15)
    Network1 = NN(n1_input,n1_hidden,n_output1,"Network1")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 2:
    #       (training 70% >fixed< ) (validation 20%)  (testing %10) 
    #       (8 inputs) (7 hidden) ( 1 output >fixed< ) 
    #       20 training iterations per epoch
    #
    #       Learning Rate = 0.5
    #       Momentum      = 0.8
    #       Weight Decay  = 0.025
    #
    (n2_input,n2_hidden) =  (8,7)
    (n_train2,n_valid2,n_test2,n_output2) = getNetworkArrays(n2_input,0.20)

    Network2 = NN(n2_input,n2_hidden,n_output2,"Network2")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 3:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (8 inputs) (6 hidden) ( 1 output >fixed< ) 
    #       10 training iterations per epoch
    #
    #       Learning Rate = 0.4
    #       Momentum      = 0.9
    #       Weight Decay  = 0.025
    #
    (n3_input,n3_hidden) =  (8,6)
    (n_train3,n_valid3,n_test3,n_output3) = getNetworkArrays(n3_input,0.15)

    Network3 = NN(n3_input,n3_hidden,n_output3,"Network3")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 4:
    #       (training 70% >fixed< ) (validation 20%)  (testing %10) 
    #       (8 inputs) (5 hidden) ( 1 output >fixed< ) 
    #       20 training iterations per epoch
    #
    #       Learning Rate = 0.4
    #       Momentum      = 0.8
    #       Weight Decay  = 0.020
    #
    (n4_input,n4_hidden) =  (8,5)
    (n_train4,n_valid4,n_test4,n_output4) = getNetworkArrays(n4_input,0.20)
    Network4 = NN(n4_input,n4_hidden,n_output4,"Network4")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 5:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (8 inputs) (4 hidden) ( 1 output >fixed< ) 
    #       10 training iterations per epoch
    #
    #       Learning Rate = 0.3
    #       Momentum      = 0.9
    #       Weight Decay  = 0.020
    #
    (n5_input,n5_hidden) =  (8,4)
    (n_train5,n_valid5,n_test5,n_output5) = getNetworkArrays(n5_input,0.15)

    Network5 = NN(n5_input,n5_hidden,n_output5,"Network5")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 6:
    #       (training 70% >fixed< ) (validation 20%)  (testing %10) 
    #       (8 inputs) (3 hidden) ( 1 output >fixed< ) 
    #       20 training iterations per epoch
    #
    #       Learning Rate = 0.3
    #       Momentum      = 0.8
    #       Weight Decay  = 0.015
    #
    (n6_input,n6_hidden) =  (8,3)
    (n_train6,n_valid6,n_test6,n_output6) = getNetworkArrays(n6_input,0.20)

    Network6 = NN(n6_input,n6_hidden,n_output6,"Network6")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Final Network:
    #       (training 70% >fixed< ) (validation 15%)  (testing %100) 
    #       (8 inputs) (6 hidden) ( 1 output >fixed< ) 
    #       15 training iterations per epoch
    #
    #       Learning Rate = 0.5
    #       Momentum      = 0.9
    #       Weight Decay  = 0.025
    #
    (f_input,f_hidden) =  (8,6)
    (n_trainF,n_validF,n_testF,n_outputF) = getNetworkArrays(f_input,0.15)
    FinalNetwork = NN(f_input,f_hidden,n_outputF,"FinalNetwork")
    
    
    while True:
        print "\n-----------------------------\n"
        print "1. Run Network"
        print "2. Load Network"
        print "3. View All Learning Rate changes"
        print "4. View Network Plots"
        print "5. Exit\n"
        print "-----------------------------\n"

        try:
            n=int(raw_input("\noption: "))
        except:
            break;
    
        if n==1:
            while True:
                try:
                    n=int(raw_input("\nWhich Network: "))
                    #Run Network 1
                    if n == 1:
                        Network1.learning_rate      = 0.5
                        Network1.momentum           = 0.9
                        Network1.weightDecayFactor  = 0.03
                        Network1.runProgram(n_train1,n_valid1,10)
                        Network1.test(n_test1)
                        saveNN(Network1,"Network1")
                        Network1.plotPrediction()
                        Network1.plotLearning()
                        Network1.plotErrors()

                    #Run Network 2
                    elif n == 2:
                        Network2.learning_rate      = 0.5
                        Network2.momentum           = 0.8
                        Network2.weightDecayFactor  = 0.025
                        Network2.runProgram(n_train2,n_valid2,20)
                        Network2.test(n_test2)
                        saveNN(Network2,"Network2")
                        Network2.plotPrediction()
                        Network2.plotLearning()
                        Network2.plotErrors()

                    #Run Network 3
                    elif n == 3:
                        Network3.learning_rate      = 0.4
                        Network3.momentum           = 0.9
                        Network3.weightDecayFactor  = 0.025
                        Network3.runProgram(n_train3,n_valid3,10)
                        Network3.test(n_test3)
                        saveNN(Network3,"Network3")
                        Network3.plotPrediction()
                        Network3.plotLearning()
                        Network3.plotErrors()

                    #Run Network 4
                    elif n == 4:
                        Network4.learning_rate      = 0.4
                        Network4.momentum           = 0.8
                        Network4.weightDecayFactor  = 0.020
                        Network4.runProgram(n_train4,n_valid4,20)
                        Network4.test(n_test4)
                        
                        saveNN(Network4,"Network4")
                        
                        Network4.plotPrediction()
                        Network4.plotLearning()
                        Network4.plotErrors()
                    
                    #Run Network 5
                    elif n == 5:
                        Network5.learning_rate      = 0.3
                        Network5.momentum           = 0.9
                        Network5.weightDecayFactor  = 0.020
                        Network5.runProgram(n_train5,n_valid5,10)
                        Network5.test(n_test5)
                        
                        saveNN(Network5,"Network5")
                        
                        Network5.plotPrediction()
                        Network5.plotLearning()
                        Network5.plotErrors()

                    #Run Network 6
                    elif n == 6:
                        Network6.learning_rate      = 0.3
                        Network6.momentum           = 0.8
                        Network6.weightDecayFactor  = 0.015
                        Network6.runProgram(n_train6,n_valid6,20)
                        Network6.test(n_test6)
                        
                        saveNN(Network6,"Network6")
                        
                        Network6.plotPrediction()
                        Network6.plotLearning()
                        Network6.plotErrors()
                    
                    elif n ==7:

                        #Test All Data in increments of 15%
                        
                        FinalNetwork.learning_rate      = 0.5
                        FinalNetwork.momentum           = 0.9
                        FinalNetwork.weightDecayFactor  = 0.025
                        FinalNetwork.runProgram(n_trainF,n_validF,15)

                        n_test1= createNormalisedDataSet(  0, 84, data,  f_input)
                        n_test2= createNormalisedDataSet(  84, 168, data,  f_input)
                        n_test3= createNormalisedDataSet( 168, 252, data,  f_input)
                        n_test4= createNormalisedDataSet(  252, 336, data,  f_input)
                        n_test5= createNormalisedDataSet(  336, 420, data,  f_input)
                        n_test6= createNormalisedDataSet(  420, 504, data,  f_input)
                        n_test7= createNormalisedDataSet(  504, 597, data,  f_input)

                        FinalNetwork.test(n_test1)
                        FinalNetwork.test(n_test2)
                        FinalNetwork.test(n_test3)
                        FinalNetwork.test(n_test4)
                        FinalNetwork.test(n_test5)
                        FinalNetwork.test(n_test6)
                        FinalNetwork.test(n_test7)

                    else:
                        break
                except:
                    break

        elif n==2:

            while True:
                #   Original Network Plotting (unsure as to working!!)
                try:
                    n=int(raw_input("Which Network: "))
                
                    if n==1:
                        print "Loaded Network 1"
                        Network1=loadNN("Network1")
                        plotNN(Network1)
                       
                    elif n==2:
                        print "Loaded Network 2"
                        Network2=loadNN("Network2")
                        plotNN(Network2)
                        
                    elif n==3:
                        print "Loaded Network 3"
                        Network3=loadNN("Network3")
                        plotNN(Network3)
                        
                    elif n==4:
                        print "Loaded Network 4"
                        Network4=loadNN("Network4")
                        plotNN(Network4)
                        
                    elif n==5:
                        print "Loaded Network 5"
                        Network5=loadNN("Network5")
                        plotNN(Network5)
                        
                    elif n==6:
                        print "Loaded Network 6"
                        Network6=loadNN("Network6")
                        plotNN(Network6)
                    elif n==7:
                        print "Loaded Final Network"
                        FinalNetwork=loadNN("Final Network")
                        plotNN( FinalNetwork)
                    else:
                        break
       
                except:
                    break
        
        elif n==3:

            NETWORKS_PLOTTING.append({"array":loadNetwork("Network1.txt")["learning"],"label":"Network1"})
            NETWORKS_PLOTTING.append({"array":loadNetwork("Network2.txt")["learning"],"label":"Network2"})
            NETWORKS_PLOTTING.append({"array":loadNetwork("Network3.txt")["learning"],"label":"Network3"})
            NETWORKS_PLOTTING.append({"array":loadNetwork("Network4.txt")["learning"],"label":"Network4"})
            NETWORKS_PLOTTING.append({"array":loadNetwork("Network5.txt")["learning"],"label":"Network5"})
            NETWORKS_PLOTTING.append({"array":loadNetwork("Network6.txt")["learning"],"label":"Network6"})

            plotNetworks(NETWORKS_PLOTTING,"Network Learning Rates","NetworksLR")
        
        elif n==4:
            n=int(raw_input("Which Network: "))
            
            #Network Plotting
            try:
                if n==1:
                    Network1.plotPrediction()
                    Network1.plotLearning()
                    Network1.plotErrors()
                   
                elif n==2:
                    Network2.plotPrediction()
                    Network2.plotLearning()
                    Network2.plotErrors()
                    
                elif n==3:
                    Network3.plotPrediction()
                    Network3.plotLearning()
                    Network3.plotErrors()
                   
                elif n==4:
                    Network4.plotPrediction()
                    Network4.plotLearning()
                    Network4.plotErrors()
                    
                elif n==5:
                    Network5.plotPrediction()
                    Network5.plotLearning()
                    Network5.plotErrors()
                   
                elif n==6:
                    Network6.plotPrediction()
                    Network6.plotLearning()
                    Network6.plotErrors()
                
                elif n==7:
                    FinalNetwork.plotPrediction()
                    FinalNetwork.plotLearning()
                    FinalNetwork.plotErrors()
                    
            except:
                print "Couldnt Plot Network"
                exit()
        else:
            break;

   
if __name__=="__main__":
    runNetwork()

