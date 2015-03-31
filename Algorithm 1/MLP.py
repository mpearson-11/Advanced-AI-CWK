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
class Layers:
    
    def __init__(self):
        
        self.layers=[]
        self.NumNeurons=0
        self.NumLayers=0
        
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
        self.weights=setWeights(self.NumNeurons+1,self.NumNeurons+1,0.0)
        
        self.changes=setWeights(self.NumNeurons+1,self.NumNeurons+1,1.0)

#===============================================================
#===============================================================
class NN:
    def __init__(self,inputs,hidden,outputs,name):
        
        self.netName=name
        self.fileName=""

        self.states={}
        self.learning_rate=0.5
        self.momentum=0.9
        self.weightDecayFactor=0.03

        #Add Input,Hidden and Output Layers
        self.Network=Layers()
        self.Network.addLayer("Input",inputs)
        self.Network.addLayer("Hidden",hidden)
        self.Network.addLayer("Output",outputs)
        self.initialiseWeights()

        self.trainingErrors=[]
        self.validationErrors=[]
        self.testErrors=[]

        self.exitCounter=0
        print name +"  Created"
#===============================================================    
    def captureState(self,epoch):
        self.states[epoch]={}
        self.states[epoch]["changes"]=self.getC()
        self.states[epoch]["training"]=self.trainingErrors
        self.states[epoch]["validation"]=self.validationErrors
        self.states[epoch]["weights"]=self.getW()

        #Only Keep 10 states (delete all others)
        if epoch >= 11:
            del self.states[epoch-11]
 #===============================================================       
    def getStateTypeAt(self,epoch,name):
        return self.states[epoch][name]
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
        
        #output=""
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tFeed Input Layer "
        #output+="\n--------------------------------------------------------------\n"
        

        for i in range(len(self.getLayerNeurons(0))):
            nodeI=self.getLayerNeurons(0)[i]
            nodeI.activation=inputs[i]
            #output+="\na(i)[{0}]: {1}".format(nodeI.n,nodeI.activation)
        #output+="\n-------------------------------------------\n"
        
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tFeed Hidden Layer "
        #output+="\n--------------------------------------------------------------\n"

        for nodeJ in self.getLayerNeurons(1):
            
            sumI=0.0
            for nodeI in self.getLayerNeurons(0):
                sumI += (self.getWeights(nodeI.n,nodeJ.n) * nodeI.activation)
                #output+="S(j)[{0}] += {1} * {2}\n".format(nodeJ.n,self.getWeights(nodeI.n,nodeJ.n),nodeI.activation)
            
            #output+="\n"
            nodeJ.setSum(sumI,self.learning_rate)

            #output+="\nS(j)[{0}]= {1}".format(nodeJ.n,nodeJ.S)
            #output+="\na(j)[{0}]= {1} + S(j)[{0}]".format(nodeJ.n,nodeJ.bias,nodeJ.n)  
            #output+="\n\na(j)[{0}]= sigmoid({1} + {2}) = {3}".format(nodeJ.n,nodeJ.bias,nodeJ.S,nodeJ.activation)
            #output+="\n--------------------------------------------------------------\n"
        
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tFeed Output Layer "
        #output+="\n--------------------------------------------------------------\n"
        
        for nodeK in self.getLayerNeurons(2):
            sumJ=0.0
            
            for nodeJ in self.getLayerNeurons(1):
                sumJ += (self.getWeights(nodeJ.n,nodeK.n) * nodeJ.activation)
                
                #output+="S(k)[{0}] += {1} * {2}\n".format(nodeK.n,self.getWeights(nodeJ.n,nodeK.n),nodeJ.activation)
            
            #output+="\n"
            nodeK.setSum(sumJ,self.learning_rate)

            #output+="\nS(k)[{0}]= {1}".format(nodeK.n,nodeK.S)
            #output+="\na(k)[{0}]= {1} + S(j)[{0}]".format(nodeK.n,nodeK.bias,nodeK.n)  
            #output+="\n\na(k)[{0}]= sigmoid({1} + {2}) = {3}".format(nodeK.n,nodeK.bias,nodeK.S,nodeK.activation)
                    

        #output+="\n--------------------------------------------------------------"
        #output+="\n--------------------------------------------------------------\n\n"
        #self.saveOutput(output)

#===============================================================
    def backPropagation(self,targets):
        
        #Back Propagate Output Layer
        #output=""
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tBackpropagate Output Layer"
        #output+="\n--------------------------------------------------------------"
        for nodeK in self.getLayerNeurons(2):
            error = targets[0] - nodeK.activation
            delta = derivative_function( nodeK.activation ) * error
            nodeK.setDelta(delta)
            
            #output+="\nz[k]={0} \n\ntarget[k]={1} \n\na[k] ={2} \n\ndelta=dv(z[k])* ( target[k] - a[k] )\n".format(nodeK.S,targets[0],nodeK.activation)
            #output+="\ndl(k)[{0}] = dv({1}) * ({2} - {3})".format(nodeK.n, nodeK.S,targets[0],nodeK.activation)
            #output+="\n= {0}".format(nodeK.delta)


        #output="\n--------------------------------------------------------------"
        #output+="\n\tUpdating Output Weights"
        #output+="\n--------------------------------------------------------------"
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
                #output+="\n\nlr = {0}\n".format(self.learning_rate)
                #output+="dl(k)[{0}] = {1}\n".format(k,nodeK.delta)
                #output+="a(j)[{0}] = {1}\n".format(j,nodeJ.activation)
                #output+="old w[{0}][{1}] = {2}\n".format(j,k,oldWeight)
                #output+="\n--------------------------------------------------------------"
                #output+="\nnew w[{0}][{1}] = {2} + ({3} * {4} * {5}) = {6}\n".format(j,k,oldWeight,\
                #    self.learning_rate,nodeK.delta,nodeJ.activation,self.getWeights(j,k))
                #output+="\n--------------------------------------------------------------"
        
        #Back Propagate Hidden Layer
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tBackpropagate Hidden Layer"
        #output+="\n--------------------------------------------------------------"
        for nodeJ in self.getLayerNeurons(1):
            error=0.0
            for nodeK in self.getLayerNeurons(2):
                error+= ( nodeK.delta * self.getWeights(nodeJ.n,nodeK.n) )

            delta = derivative_function( nodeJ.activation  ) * error
            nodeJ.setDelta(delta)
            #output+="\nz[j]={0} \nerror = sigma( delta[k] * w[j][k] ) = {1} \ndelta = dv(z[j]) * error \n".format(nodeJ.S,error)
            #output+="\ndl(j)[{0}] = dv({1}) * {2}".format(nodeJ.n,nodeJ.S,error)
            #output+="\n= {0}".format(nodeJ.delta)

        #Update Hidden Layer
        #output+="\n--------------------------------------------------------------"
        #output+="\n\tUpdating Hidden Weights"
        #output+="\n--------------------------------------------------------------"
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

                #output+="\n\nlr = {0}\n".format(self.learning_rate)
                #output+="dl(j)[{0}] = {1}\n".format(j,nodeJ.delta)
                #output+="a(i)[{0}] = {1}\n".format(i,nodeI.activation)
                #output+="old w[{0}][{1}] = {2}\n".format(i,j,oldWeight)
                #output+="\nold change ={0}\n".format(change)
                #output+="\nnew change ={0}\n".format(oldChange)
                #output+="\n--------------------------------------------------------------"
                #output+="\nnew w[{0}][{1}] = {2} + ({3} * {4} * {5}) = {6}\n".format(i,j,oldWeight,\
                #    self.learning_rate,nodeJ.delta,nodeI.activation,self.getWeights(i,j))
                #output+="\n--------------------------------------------------------------"

        #output+="\n------------------------------------\n"
        #output+="\n--------------------------------------------------------------"
        #output+="\n--------------------------------------------------------------\n\n"
        #self.saveOutput(output)
     
#===============================================================
    def decayWeights(self,epoch):
        for nodeK in self.getLayerNeurons(2):
            for nodeJ in self.getLayerNeurons(1):
                k=nodeK.n 
                j=nodeJ.n 
                self.getC()[j][k] -= (self.weightDecayFactor *  self.getStateTypeAt(epoch-1,"changes")[j][k] )


        for nodeJ in self.getLayerNeurons(1):
            for nodeI in self.getLayerNeurons(0):
                j=nodeJ.n 
                i=nodeI.n 
                self.getC()[i][j] -= (self.weightDecayFactor *  self.getStateTypeAt(epoch-1,"changes")[i][j] ) 
#===============================================================
    def train(self,examples,iterations):

        Errors=[]
        
        for i in range(iterations):
            error=0.0
            for values in examples:
                
                inputs  = values[0]
                outputs = values[1]

                self.feed_forward(inputs)
                self.backPropagation(outputs)

                error += self.getError(outputs)

        error = math.sqrt( error / 2.0 )
        Errors.append(error)
        return Errors

#===============================================================     
    def runProgram(self,TRAIN,VALID,n):
        
        print "Running : "+self.netName
        oldError=0.0
        error=0.0
        epoch=0

        while True:

            #Train with n iterations
            self.trainingErrors=self.train(TRAIN,n)
            self.validationErrors=[]

            oldError=error
            error=0.0

            for values in VALID:
                
                inputs  = values[0]
                outputs = values[1]

                self.feed_forward(inputs)
                self.backPropagation(outputs)

                error += self.getError(outputs)

            error = math.sqrt( error / 2.0 )
            self.validationErrors.append(error)

            self.bold_driver(error,oldError)
            
            #Capture state to allow weight decay access to weight changes by epoch
            self.captureState(epoch)
            
            if epoch > 1:
                self.decayWeights(epoch)
            
            print "{0} Error = {1} LR = {2}".format(epoch,error,self.learning_rate)
           
            #checkCount updates termination value
            self.checkCount(error,oldError)

            if self.exitCounter > 10 or epoch > 500:
                #Go back 10 to minimum epoch/time 

                epochMinima = epoch - 10
                
                #Override all Network Weights for Best test data 
                #based on minima termination weights
                self.validationErrors    = self.getStateTypeAt(epochMinima, 'validation' )
                self.trainingErrors      = self.getStateTypeAt(epochMinima, 'training'   )
                self.Network.weights     = self.getStateTypeAt(epochMinima, 'weights'    )
                self.Network.changes     = self.getStateTypeAt(epochMinima, 'changes'    )

                del self.states

                print "\n\n------------------------------------------"
                print "\nEpochs: "+str(epoch)
                print "\nExit Counter: "+str(self.exitCounter)
                print "\n------------------------------------------"
                break
            
            else:
                epoch+=1

#===============================================================
    def test(self,examples):

        predictions=[]
        actualValues=[]

        for values in examples:
            self.feed_forward(values[0])
            actual=values[1]
            predictor=self.getPrediction()
            print "Actual : {0} Prediction: {1}".format(actual[0],predictor)
            predictions.append(predictor)
            actualValues.append(actual[0])

        smooth_plot(actualValues,predictions,"Predictions","output (red), prediction (blue)")
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
    def checkCount(self,new,old):
        if (new < old):
            self.exitCounter=0

        else:
            self.exitCounter+=1   
#===============================================================
    def plotAll(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        #Training Plot
        x1 =np.array( vector( len(self.trainingErrors) ) )
        y1 = np.array(self.trainingErrors)

        x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
        y_smooth1 = spline(x1, y1, x_smooth1)

        #Validation Plot
        x2 =np.array( vector( len(self.validationErrors) ) )
        y2= np.array(self.validationErrors)

        x_smooth2 = np.linspace(x2.min(), x2.max(), 200)
        y_smooth2 = spline(x2, y2, x_smooth2)

      
        ax.plot(x_smooth1,y_smooth1,'r')
        ax.plot(x_smooth2,y_smooth2,'g')

        plt.title("Red = Train, Green = Validate")
        fig.savefig("Plot.pdf")

#===============================================================
#===============================================================




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

def runNetwork(n):
    

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 1:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (8 inputs) (8 hidden) ( 1 output >fixed< ) 
    #       20 training iterations per epoch
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
    #       (8 inputs) (4 hidden) ( 1 output >fixed< ) 
    #       18 training iterations per epoch
    #
    #       Learning Rate = 0.5
    #       Momentum      = 0.8
    #       Weight Decay  = 0.025
    #
    (n2_input,n2_hidden) =  (8,4)
    (n_train2,n_valid2,n_test2,n_output2) = getNetworkArrays(n2_input,0.20)

    Network2 = NN(n2_input,n2_hidden,n_output2,"Network2")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 3:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (6 inputs) (8 hidden) ( 1 output >fixed< ) 
    #       16 training iterations per epoch
    #
    #       Learning Rate = 0.4
    #       Momentum      = 0.9
    #       Weight Decay  = 0.025
    #
    (n3_input,n3_hidden) =  (6,8)
    (n_train3,n_valid3,n_test3,n_output3) = getNetworkArrays(n3_input,0.15)

    Network3 = NN(n3_input,n3_hidden,n_output3,"Network3")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 4:
    #       (training 70% >fixed< ) (validation 20%)  (testing %10) 
    #       (6 inputs) (4 hidden) ( 1 output >fixed< ) 
    #       14 training iterations per epoch
    #
    #       Learning Rate = 0.4
    #       Momentum      = 0.8
    #       Weight Decay  = 0.020
    #
    (n4_input,n4_hidden) =  (6,4)
    (n_train4,n_valid4,n_test4,n_output4) = getNetworkArrays(n4_input,0.20)
    Network4 = NN(n4_input,n4_hidden,n_output4,"Network4")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 5:
    #       (training 70% >fixed< ) (validation 15%)  (testing %15) 
    #       (4 inputs) (8 hidden) ( 1 output >fixed< ) 
    #       16 training iterations per epoch
    #
    #       Learning Rate = 0.3
    #       Momentum      = 0.9
    #       Weight Decay  = 0.020
    #
    (n5_input,n5_hidden) =  (4,8)
    (n_train5,n_valid5,n_test5,n_output5) = getNetworkArrays(n5_input,0.15)

    Network5 = NN(n5_input,n5_hidden,n_output5,"Network5")

    #-----------------------------------------------------------------------
    #-----------------------------------------------------------------------
    #   Network 6:
    #       (training 70% >fixed< ) (validation 20%)  (testing %10) 
    #       (4 inputs) (4 hidden) ( 1 output >fixed< ) 
    #       16 training iterations per epoch
    #
    #       Learning Rate = 0.3
    #       Momentum      = 0.8
    #       Weight Decay  = 0.015
    #
    (n6_input,n6_hidden) =  (4,4)
    (n_train6,n_valid6,n_test6,n_output6) = getNetworkArrays(n6_input,0.20)

    Network6 = NN(n6_input,n6_hidden,n_output6,"Network6")

    #Run Network 1
    if n == 1:
        Network1.learning_rate      = 0.5
        Network1.momentum           = 0.9
        Network1.weightDecayFactor  = 0.03

        Network1.runProgram(n_train1,n_valid1,20)
        Network1.test(n_test1)

    
    #Run Network 2
    elif n == 2:
        Network2.learning_rate      = 0.5
        Network2.momentum           = 0.8
        Network2.weightDecayFactor  = 0.025

        Network2.runProgram(n_train2,n_valid2,18)
        Network2.test(n_test2)

    
    #Run Network 3
    elif n == 3:
        Network3.learning_rate      = 0.4
        Network3.momentum           = 0.9
        Network3.weightDecayFactor  = 0.025

        Network3.runProgram(n_train3,n_valid3,16)
        Network3.test(n_test3)


    #Run Network 4
    elif n == 4:
        Network4.learning_rate      = 0.4
        Network4.momentum           = 0.8
        Network4.weightDecayFactor  = 0.020

        Network4.runProgram(n_train4,n_valid4,14)
        Network4.test(n_test4)

    
    #Run Network 5
    elif n == 5:
        Network5.learning_rate      = 0.3
        Network5.momentum           = 0.9
        Network5.weightDecayFactor  = 0.020

        Network5.runProgram(n_train5,n_valid5,12)
        Network5.test(n_test5)


    #Run Network 6
    elif n == 6:
        Network6.learning_rate      = 0.3
        Network6.momentum           = 0.8
        Network6.weightDecayFactor  = 0.015

        Network6.runProgram(n_train6,n_valid6,10)
        Network6.test(n_test6)




if __name__=="__main__":
    option=int(raw_input("Network 1,2,3,4,5,6? "))
    runNetwork(option)

