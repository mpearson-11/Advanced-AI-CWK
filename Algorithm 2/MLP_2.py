#   @Copywright Max Pearson
#   Student ID: B123103
#   Date Created 01/03/2015
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
from copy import *

#--------------------------------------------------------------------------
#Data Class for Neural Network use
from Data import *
from Neurons import *

#--------------------------------------------------------------------------
#Scientific Packages for plotting and array manipulation
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

#Package to neatly tabulate 2 Dimensional arrays
from tabulate import *

###########################################################################
# Populate an array with indexes (graphing)
###########################################################################

def vector(n):
    vector = []
    for i in range(n):
        vector.append(i)
    return vector

###########################################################################
# Populate 2d array with length outerNum and inner Length innerNum
###########################################################################

def populateVector(outerNum, innerNum):
    vector = []
    for i in range(outerNum):
        vector.append([0.0]*innerNum)
    return vector

###########################################################################
# Hyperbolic tangent functions
########################################################################### 

def hyperbolic_tangent(n):
    return math.tanh(n)

def hyperbolic_tangent_dv(n):
    return (1.0 - math.pow(n,2))

###########################################################################
# Loigistic functions
###########################################################################

def sigmoid(n):
    return 1.0 / (1.0 + math.exp(-n))

def sigmoid_dv(n):
    return n * ( 1.0 - n )

#Activation
def activation_function(n):
    return hyperbolic_tangent(n)
    #return sigmoid(n)

def derivative_function(n):
    return hyperbolic_tangent_dv(n)
    #return sigmoid_dv(n)

#--------------------------------------------------------------------------
#Initiate Random Generator
random.seed(0)
#--------------------------------------------------------------------------
###########################################################################
#                         ------------------------
#                         |     NETWORK_ class    |
#                         ------------------------
###########################################################################
class NETWORK_:
    def __init__(self):
        pass
###########################################################################
#                         ------------------------
#                         |     NETWORK class    |
#                         ------------------------
###########################################################################

class MLP:
    def __init__(self):

        self.__learning_rate = 0.5
        self.__momentum      = 0.9  
        self.__Neurons=Neurons()

    #Clone Network instance
    def cloneState(self):
        #Create Neuron from clone and delete once allocated
        networkClone=NETWORK_()
        
        networkClone.learning_rate      = self.__learning_rate
        networkClone.momentum           = self.__momentum
        
        networkClone.IH_WEIGHTS         = self.neuron.getIHArray()
        networkClone.HO_WEIGHTS         = self.neuron.getHOArray()
        
        networkClone.input_activation   = self.neuron.getInputActivationArray()
        networkClone.output_activation  = self.neuron.getOutputActivationArray()
        networkClone.hidden_activation  = self.neuron.getHiddenActivationArray()
        
        networkClone.input_change       = self.neuron.getInputChangeArray()
        networkClone.output_change      = self.neuron.getOutputChangeArray()

        return networkClone

    def startPerceptronLearning(self):
        self.runTraining(500)
        self.runValidation(1000)
        self.runTestOn(self.__testingData,"Testing")

    def assignTrainingData(self,Data):
        self.__trainingData=Data
        self.createNetwork()

    def assignValidationData(self,Data):
        self.__validationData=Data

    def assignTestingData(self,Data):
        self.__testingData=Data

    def createNetwork(self):

        self.__number_of_inputs  = len(self.__trainingData[0][0]) + 1    # 1 more for bias 
        self.__number_of_hidden  = len(self.__trainingData[0][0])
        self.__number_of_outputs = len(self.__trainingData[0][1])

        self.neuron = NETWORK(  self.__number_of_inputs,\
                                self.__number_of_hidden,\
                                self.__number_of_outputs)

    def getInputSize(self):
        return self.__number_of_inputs

    def getHiddenSize(self):
        return self.__number_of_hidden

    def getOutputSize(self):
        return self.__number_of_outputs
  

    def InitialiseWeights(self):

        for i in range(self.__number_of_inputs):
            for j in range(self.__number_of_hidden):
                self.neuron.setIH( i , j , generateIH(self) )

        for j in range(self.__number_of_hidden):
            for k in range(self.__number_of_outputs):
                self.neuron.setHO( j , k , generateHO(self) )

    def feed_forward(self, inputs):
        #----------------------------------------------------------------------

        
        #Input Activations (-1 for lack of bias)
        for i in range((self.__number_of_inputs-1)):
            self.neuron.setInputActivation(i,inputs[i])
        
        #----------------------------------------------------------------------
        #Hidden Activations
        for j in range(self.__number_of_hidden):
            sumOfHidden = 0.0
            
            for i in range(self.__number_of_inputs):
                
                sumOfHidden  = sumOfHidden  \
                + self.neuron.getInputActivation(i) * self.neuron.getIH(i , j)
                                        
            self.neuron.setHiddenActivation(j , activation_function(sumOfHidden) )
        #----------------------------------------------------------------------
        #Output Activations
        for k in range(self.__number_of_outputs):
            sumOfOutput = 0.0
            for j in range(self.__number_of_hidden):
                
                sumOfOutput = sumOfOutput \
                + self.neuron.getHiddenActivation(j) * self.neuron.getHO( j , k )
            
            self.neuron.setOutputActivation(k, activation_function(sumOfOutput) )

        #----------------------------------------------------------------------
        return self.neuron.getOutputActivationArray()
    
    def calculate_output_error(self,targets,outputDELTA):
        #----------------------------------------------------------------------
        for k in range(self.__number_of_outputs):
            error = targets[k] - self.neuron.getOutputActivation(k)
            outputDELTA[k] = derivative_function(self.neuron.getOutputActivation(k)) * error
        #----------------------------------------------------------------------
        return outputDELTA
    
    def calculate_hidden_error(self,hiddenDELTA,outputDELTA):
        #----------------------------------------------------------------------
        for j in range(self.__number_of_hidden):
            error = 0.0
            for k in range(self.__number_of_outputs):
                error = error + (outputDELTA[k] * self.neuron.getHO(j,k) )
            
            hiddenDELTA[j] =  derivative_function(self.neuron.getHiddenActivation(j)) * error
    
        #----------------------------------------------------------------------
        return hiddenDELTA

    def bold_driver(self,ERRORS,valNum):
        size=len(ERRORS)

        if size > 1:
            
            new=ERRORS[size-1]
            old=ERRORS[size-2]
            
            if new < old:
                if self.__learning_rate > 0.9:
                    pass
                else:
                    self.__learning_rate *= 1.01
                valNum=0
                    
            elif new > old:
                self.__learning_rate *= 0.50
                valNum+=1

            else:
                pass

        return valNum
        

    def update_HO(self,outputDELTA):
       
        for j in range(self.__number_of_hidden):
            for k in range(self.__number_of_outputs):
                change = outputDELTA[k] * self.neuron.getHiddenActivation(j)
                
                output= self.neuron.getHO(j,k)\
                + (self.__learning_rate  * change) \
                + (self.__momentum       * self.neuron.getOutputChange(j,k) )
                
                self.neuron.setHO(j,k,output)
                self.neuron.setOutputChange(j,k,change)
    
    def update_IH(self,hiddenDELTA):
        
        #----------------------------------------------------------------------
        
        for i in range(self.__number_of_inputs):
            for j in range(self.__number_of_hidden):
                change = hiddenDELTA[j]*self.neuron.getInputActivation(i)
                

                output = self.neuron.getIH(i,j)\
                + (self.__learning_rate  * change) \
                + (self.__momentum       * self.neuron.getInputChange(i,j) )

                self.neuron.setIH(i,j,output)
                self.neuron.setInputChange(i,j,change)

        #----------------------------------------------------------------------
    
    
    def backPropagate(self, targets):

        if len(targets) != self.__number_of_outputs:
            print "Cannot create Outputs, targets size doesnt equal outputs"
            exit()

        outputDELTA = [0.0] * self.__number_of_outputs
        outputDELTA = self.calculate_output_error(targets,outputDELTA)

        hiddenDELTA = [0.0] * self.__number_of_hidden
        hiddenDELTA = self.calculate_hidden_error(hiddenDELTA,outputDELTA)

        self.update_HO(outputDELTA)
        self.update_IH(hiddenDELTA)

        error = 0.0
        
        for k in range(len(targets)):
            sq_rtError = targets[k] - self.neuron.getOutputActivation(k)
            error += math.pow(sq_rtError,2)
       
        return error / len(targets)
    
    ###########################################################################
    #   With Hidden and Output Wieghts set and errors found
    #   run test data through forward pass function to output predictions.
    ###########################################################################
   
    def runTestOn(self, examples, name):
        #----------------------------------------------------------------------
        predictions=[]
        actual=[]
        node=0
        for inputObj in examples:

            inputNodes=inputObj[0]
            predictionForNode=self.feed_forward( inputNodes )
            actualValues             = inputObj[1]
            predictions.append( predictionForNode )            
            actual.append( actualValues )
        #----------------------------------------------------------------------
        #Plot Actual against predictions
        smooth_plot( actual , predictions, name )
        #----------------------------------------------------------------------

    ###########################################################################
    # Run Training input into backpropagation algorithm to find error 
    # calculations.
    ###########################################################################

    def runValidation(self, epochs):
    
        examples=self.__validationData

        print "________________________________________________"
        print "\t# Validating Network\n"
        print "________________________________________________"
        print "\n\t(epoch number) := (error)\n"

        ERRORS=[]
        validationCounter=0

        for epoch in range(epochs):

            error = 0.0
            
            for obj in examples:
                inputs  = obj[0]          
                targets = obj[1]                    
                self.feed_forward(inputs)             
                RMSE    = self.backPropagate(targets)
                error   += RMSE
        
            ERRORS.append(error)
            validationCounter=self.bold_driver(ERRORS,validationCounter)

            networkClone=self.cloneState()
            self.__Neurons.addNeuron(epoch,networkClone,ERRORS[epoch])
            del networkClone
            
            if validationCounter >= 10:
                print "\n\t# Finished Validation\n\n"
                selectedNeuron = ( epoch - 10 )       
                self.__Neurons.setSelectedNeuron(selectedNeuron)
                outputThisNeuron = self.__Neurons.getNeuron(selectedNeuron)
                
                #Pass Neuron through save errors
                save_errors(ERRORS,1)
                #Show Neuron at epoch selected with WEIGHTS AND BIASES
                show(outputThisNeuron)
                return
            else:
                if epoch == epochs-1:
                    print "Terminating - (Minima was not found)"
                    return 

            print "\t({0}) := ({1}) lr: ={2}"\
            .format(epoch,error,self.__learning_rate)

        #Save Errors to File
        save_errors(ERRORS,1)
    
    def runTraining(self, epochs):

        examples=self.__trainingData
    
        #----------------------------------------------------------------------
        # Main Execution of Training (print errors) 
        print "________________________________________________"
        print "\t# Training Network\n"
        print "________________________________________________"
        print "\n\t(epoch number) := (error)\n"

        ERRORS=[]
        for epoch in range(epochs):
            error = 0.0
            for obj in examples:

                inputs = obj[0]          
                targets = obj[1]                     
                self.feed_forward(inputs)             
                RMSE=self.backPropagate(targets)
                error  += RMSE
                
            ERRORS.append(error)
            valNum=self.bold_driver(ERRORS,0)            
            del valNum

            print "\t({0}) := ({1})"\
            .format(epoch,error)

        #Save Errors to File
        save_errors(ERRORS,0)



class NETWORK:

    def __init__(self,n1,n2,n3):

        self.__number_of_inputs  = n1 
        self.__number_of_hidden  = n2
        self.__number_of_outputs = n3
                   
        self.__input_activation  = [1.0] * n1
        self.__hidden_activation = [1.0] * n2
        self.__output_activation = [1.0] * n3
        
        self.__IH_WEIGHTS    = populateVector(n1,n2)
        self.__HO_WEIGHTS    = populateVector(n2,n3)
  
        self.__input_change  = populateVector(n1,n2)
        self.__output_change = populateVector(n2,n3)
    
    #Hidden and Output Layer Weight functions
    def getHO(self,i1,i2):
        return self.__HO_WEIGHTS[i1][i2]
    
    def getHOArray(self):
        return self.__HO_WEIGHTS

    def setHO(self,i1,i2,VAL):
        self.__HO_WEIGHTS[i1][i2]=VAL

    def getIH(self,i1,i2):
        return self.__IH_WEIGHTS[i1][i2]

    def getIHArray(self):
        return self.__IH_WEIGHTS
        
    def setIH(self,i1,i2,VAL):
        self.__IH_WEIGHTS[i1][i2]=VAL

    #Input Activation Functions
    def getInputActivation(self,i1):
        return self.__input_activation[i1]

    def getInputActivationArray(self):
        return self.__input_activation

    def setInputActivation(self,i1,VAL):
        self.__input_activation[i1]=VAL

    #Output Activation Functions
    def getOutputActivation(self,i1):
        return self.__output_activation[i1]

    def getOutputActivationArray(self):
        return self.__output_activation

    def setOutputActivation(self,i1,VAL):
        self.__output_activation[i1]=VAL

    #Hidden Activation Functions
    def getHiddenActivation(self,i1):
        return self.__hidden_activation[i1]

        #Hidden Activation Functions
    def getHiddenActivationArray(self):
        return self.__hidden_activation

    def setHiddenActivation(self,i1,VAL):
        self.__hidden_activation[i1]=VAL

    #Input Change Functions
    def getInputChange(self,i1,i2):
        return self.__input_change[i1][i2]

    #Input Change Functions
    def getInputChangeArray(self):
        return self.__input_change

    def setInputChange(self,i1,i2,VAL):
        self.__input_change[i1][i2]=VAL

    #Input Change Functions
    def getOutputChange(self,i1,i2):
        return self.__output_change[i1][i2]

    #Input Change Functions
    def getOutputChangeArray(self):
        return self.__output_change

    def setOutputChange(self,i1,i2,VAL):
        self.__output_change[i1][i2]=VAL

    

###########################################################################
#   Save RMS Errors to File errors.txt
###########################################################################    
def save_errors(errors,TYPE):
    
    if TYPE==0:
        fileName="TrainingErrors.txt"
        name="# Training Errors"
    
    elif TYPE==1:
        fileName="ValidationErrors.txt"
        name="# Validation Errors"
    
    #Now created rewrite
    errorsFile= open(fileName, "w")

    output=""
    output+="\n----------------------------------------\n"
    output+=name
    output+="\n----------------------------------------\n"

    for i in range(len(errors)):
        output+=("epoch: "+str(i)+" = "+str(errors[i]))
        output+="\n"

    errorsFile.write(output)
    errorsFile.close()
    plot_errors(errors,TYPE)

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

    fig.savefig(name+'_Predictions.pdf')
    


def plot_errors(error,TYPE):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #----------------------------------------------------------------------
    y1=np.array(error)
    #----------------------------------------------------------------------
  
    if TYPE == 0:
        ax.plot(y1,'r')
        fig.savefig('TrainingSetErrors.pdf')
    
    elif TYPE == 1:
        ax.plot(y1,'g')
        fig.savefig('ValidationSetErrors.pdf')
    
        

def generateIH(network):
    
    a = -2.0 / network.getInputSize()
    b = 2.0 / network.getInputSize()
    number = (b - a) * random.random() + a
    return number

def generateHO(network):
   
    a = -2.0 / network.getHiddenSize()
    b = 2.0 / network.getHiddenSize()
    number = (b - a) * random.random() + a
    return number

def show(neuron):
    epoch=neuron.index

    neuronNetwork=neuron.getNetwork()
    
    fileName="Details_Epoch_"+str(epoch)+".txt"
    weightsFile= open(fileName, "w")
    weightsFile.close()

    #Now created rewrite
    weightsFile= open(fileName, "w")

    output="\nFinal Epoch = "
    output+=str(epoch)
    output+="\nError = "
    output+=neuron.getError()
    output+="\n-------------------------\n"
    output+="Final Learning Rate: "
    output+=str(neuronNetwork.learning_rate)
    output+="\n"
    output+="Final Momentum: "
    output+=str(neuronNetwork.momentum)
    output+="\n"
    output+="-------------------------\n"
    output+="# IH Weights"
    output+="\n-------------------------\n"
    output += tabulate( neuronNetwork.IH_WEIGHTS)
    
    output+="\n-------------------------\n"
    output+="# HO Weights"
    output+="\n-------------------------\n"
    output += tabulate( neuronNetwork.HO_WEIGHTS)
    
    
    output+="\n-------------------------\n"
    output+="# Input Final Changes"
    output+="\n-------------------------\n"
    output+= tabulate( neuronNetwork.input_change)
    
    
    output+="\n-------------------------\n"
    output+="# Output Final Changes"
    output+="\n-------------------------\n"
    output+= tabulate( neuronNetwork.output_change)

    output+="\n-------------------------\n"
    output+="# Input Activations"
    output+="\n-------------------------\n"
    

    output+="---------------------\n"
    for i in neuronNetwork.input_activation:
        output+=str(i)
        output+="\n"
    output+="\n---------------------\n"
    
    output+="\n-------------------------\n"
    output+="# Output Activations"
    output+="\n-------------------------\n"
    
    output+="---------------------\n"
    for i in neuronNetwork.output_activation:
        output+=str(i)
        output+="\n"
    output+="\n---------------------\n"

    output+="\n-------------------------\n"
    output+="# Hidden Activations"
    output+="\n-------------------------\n"
    
    output+="---------------------\n"
    for i in neuronNetwork.hidden_activation:
        output+=str(i)
        output+="\n"
    output+="\n---------------------\n"
    
    weightsFile.write(output)
    weightsFile.close()


###########################################################################
#                       Initiate Program 
###########################################################################
def execute_MLP():
    #----------------------------------------------------------------------
    # Call class Data (populate data structure)

    data = Data()
    #----------------------------------------------------------------------
    # Create Training and Test Data from CWKData.xlsx
    # Take 80% of data (400 epochs as default) and leave rest for testing
    
    DATA_SET        =   400
    INPUTS          =   8

    #Data Variables
    T_START         =   1
    T_END           =   T_START    + int(DATA_SET * 0.7)
    
    TTS_START       =   T_END
    TTS_END         =   TTS_START  + int(DATA_SET * 0.15)
    
    TST_START       =   TTS_END
    TST_END         =   TST_START  + int(DATA_SET * 0.15)    


    #CREATE NORMALISED DATA FROM FILE
    TRAINING_DATA   = createNormalisedDataSet(  T_START,   T_END,   data,  INPUTS)
    VALIDATION_DATA = createNormalisedDataSet(  TTS_START, TTS_END, data,  INPUTS)
    TESTING_DATA    = createNormalisedDataSet(  TST_START, TST_END, data,  INPUTS)

    
    MultilayerPerceptron   = MLP ()    
    MultilayerPerceptron.assignTrainingData(TRAINING_DATA)
    MultilayerPerceptron.assignValidationData(VALIDATION_DATA)
    MultilayerPerceptron.assignTestingData(TESTING_DATA)
    MultilayerPerceptron.InitialiseWeights()
    MultilayerPerceptron.startPerceptronLearning()

execute_MLP()
   

