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
from random import *
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


#Figure for plotting Errors
ERROR_FIGURE=plt.figure()

seed(0)
#--------------------------------------------------------------------------

def checkErrors(errors):
    size=len(errors)-1
    if errors[size] > errors[size-1]:
        return True
    else:
        return False 

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
    #return hyperbolic_tangent(n)
    return sigmoid(n)

def derivative_function(n):
    #return hyperbolic_tangent_dv(n)
    return sigmoid_dv(n)

def sigma(w,u,j):
    size=len(u)
    
    sumOf=0.0
    
    for i in range(size):
        sumOf +=  (w[i][j] * u[i])

    return sumOf

def anneal(x,r):
        bottom = 1.0 + math.exp(10.0 - (20.0 * x / r) ) 
        return 1 - (1.0 / bottom)
#--------------------------------------------------------------------------
#Initiate Random Generator

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




class NETWORK:

    def cloneState(self):
        #Create Neuron from clone and delete once allocated
        networkClone=NETWORK_()
        
        networkClone.learning_rate      = self.learning_rate
        networkClone.momentum           = self.momentum
        
        networkClone.HI_WEIGHTS         = self.HI_WEIGHTS
        networkClone.HO_WEIGHTS         = self.HO_WEIGHTS
        
        networkClone.input_activation   = self.input_activation
        networkClone.output_activation  = self.output_activation
        networkClone.hidden_activation  = self.hidden_activation
        
        networkClone.input_change       = self.input_change
        networkClone.output_change      = self.output_change

        networkClone.trainingID         = self.trainingID

        return networkClone

    def getNeurons(self,TYPE):
    
        if TYPE==0:
            return self.__trainingNeurons

        elif TYPE==0:
            return self.__validationNeurons
    
    def __init__(self, number_of_inputs, number_of_hidden, \
                       number_of_outputs):
        
        self.__validationNeurons=Neurons()
        self.__trainingNeurons=Neurons()

        self.trainingID=0
        #------------------------------------------------------------------
        #   Neural Network Settings

        self.number_of_inputs  = number_of_inputs + 1    # 1 more for bias 

        self.number_of_hidden , self.number_of_outputs = (\
        number_of_hidden      , number_of_outputs        ) 

        
        
        #Default Learning Rate and Momentum (subject to change)
        self.learning_rate , self.momentum = ( \
        0.5               , 0.9              )

        #------------------------------------------------------------------
        # Bias Activations
        n=[1.0]
        
        (   self.input_activation,  \
            self.hidden_activation, \
            self.output_activation  ) = ( n * self.number_of_inputs,\
                                          n * self.number_of_hidden,\
                                          n * self.number_of_outputs )
        #------------------------------------------------------------------
        
        # Hidden Input and Hidden Output Weights
        
        self.HI_WEIGHTS    = populateVector(self.number_of_inputs,\
                                            self.number_of_hidden)

        self.HO_WEIGHTS    = populateVector(self.number_of_hidden,\
                                            self.number_of_outputs)

        # Final Values for Change in weights for Momentum   
        self.input_change  = populateVector(self.number_of_inputs,\
                                            self.number_of_hidden)

        self.output_change = populateVector(self.number_of_hidden,\
                                            self.number_of_outputs)

        self.outputDELTA = [0.0] * self.number_of_outputs
        self.hiddenDELTA = [0.0] * self.number_of_hidden

        self.sum_hidden= [0.0] * self.number_of_hidden
        self.sum_output = [0.0] * self.number_of_outputs

        #----------------------------------------------------------------------
        #Print formatted Network Configuration
        print "________________________________________________"
        print "\tNeural Network\n"
        print "________________________________________________"
        print "\tHidden = {0}\n\tOutputs = {1}\n\tInputs = {2} + 1 bias"\
        .format(self.number_of_hidden,\
                    self.number_of_outputs, \
                    self.number_of_inputs-1)

        print "\tLearning Rate = {0}\n\tMomentum = {1}"\
        .format(    self.learning_rate, \
                    self.momentum           )        
        #----------------------------------------------------------------------
        # Random Weight Assigment
        
        print "\nWeights Initialised\n"
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden):
                self.HI_WEIGHTS[i][j] = generateRandFor(self,"HI")

        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                self.HO_WEIGHTS[j][k] = generateRandFor(self,"HO")
        #----------------------------------------------------------------------

    ###########################################################################
    # Run FeedForward function for (testing and training) setting 
    # hidden activations and return output activations
    ###########################################################################

    


    def feed_forward(self, inputs):
        #----------------------------------------------------------------------

        #Input Activations (-1 for lack of bias)
        for i in range(self.number_of_inputs-1):
            self.input_activation[i] = inputs[i]
        #----------------------------------------------------------------------
        #Hidden Activations
        #print "\nFeed Input\n"
        for j in range(self.number_of_hidden):
            sumOfHidden = sigma(self.HI_WEIGHTS,self.input_activation,j)
            self.hidden_activation[j] = activation_function(sumOfHidden)

            #print "j : {0} | S_j :{1} | f(S_j) = {2}  ".format(j,sumOfHidden,self.hidden_activation[j])
        #print "\nFeed Hidden Activations\n"
        for k in range(self.number_of_outputs):
            sumOfOutput = sigma(self.HO_WEIGHTS,self.hidden_activation,k)
            self.sum_output[k]=sumOfOutput
            self.output_activation[k] = activation_function(sumOfOutput)
            #print "k : {0} | S_k :{1} | f(S_k) = {2}  ".format(k,sumOfOutput,self.output_activation[k])
        #----------------------------------------------------------------------

    def calculate_OutputDelta(self,targets):
        #----------------------------------------------------------------------
        for k in range(self.number_of_outputs):
            error = targets[k] - self.output_activation[k]
            self.outputDELTA[k] = derivative_function(self.output_activation[k]) * error
        #----------------------------------------------------------------------
        
    def calculate_HiddenDelta(self):
        #----------------------------------------------------------------------
        for j in range(self.number_of_hidden):
            error = 0.0
            for k in range(self.number_of_outputs):
                error = error + (self.outputDELTA[k] * self.HO_WEIGHTS[j][k])
            
            self.hiddenDELTA[j] =  derivative_function(self.hidden_activation[j]) * error      
        #----------------------------------------------------------------------

    def weightDecay(self,n,weight):
        v = 1 / self.learning_rate * n
        
        sumOf=0.0
        for i in range(len(weight)):
            sumOf+=math.pow(weight[i],2)

        omega=sumOf / 2.0

        return np.array(weight) + ( v * omega ) 

    def simulated_annealing(self,error,totalEpochs):
        p=0.01
        q=self.learning_rate
        r=totalEpochs
        x=error
        return p + ((q - p) * anneal(x,r) )

    def bold_driver(self,ERRORS,valNum):
        size=len(ERRORS)
        inc=1.01
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
        
    ###########################################################################
    # Update Hiddden Output weights
    ###########################################################################
    def update_HiddenOutput(self):
        #----------------------------------------------------------------------
       
        for j in range(self.number_of_hidden):
            
            for k in range(self.number_of_outputs):
                change = self.outputDELTA[k] * self.hidden_activation[j]

                oldWeight=self.HO_WEIGHTS[j][k]
                
                self.HO_WEIGHTS[j][k] = oldWeight \
                + (self.learning_rate  * change) \
                + (self.momentum       * self.output_change[j][k])
                
                self.output_change[j][k] = change
                

        #----------------------------------------------------------------------
    
    ###########################################################################
    # Update Input Hiddden weights
    ###########################################################################
    def update_HiddenInput(self):
        
        #----------------------------------------------------------------------
        
        for i in range(self.number_of_inputs-1):
            
            for j in range(self.number_of_hidden):
                
                change = self.hiddenDELTA[j] * self.input_activation[i]
                
                oldWeight=self.HI_WEIGHTS[i][j]

                self.HI_WEIGHTS[i][j] = oldWeight\
                + (self.learning_rate  * change) \
                + (self.momentum       * self.input_change[i][j])


                
                self.input_change[i][j] = change
        #----------------------------------------------------------------------
    
    def updateWeights(self,epoch):

        
        #Update Output Hidden Weights
        self.update_HiddenOutput()

        #Update Input Hidden Weights
        self.update_HiddenInput()

        #----------------------------------------------------------------------
        



    def calculateError(self,targets):
        errors=0.0

        for i in range(self.number_of_outputs):
            inner = math.pow(targets[i]-self.output_activation[i],2)
            errors += inner

        return math.sqrt(errors/len(targets))

    
    def backPropagate(self, targets):
        #----------------------------------------------------------------------
        if len(targets) != self.number_of_outputs:
            print "Cannot create Outputs, targets size doesnt equal outputs"
            exit()
        #----------------------------------------------------------------------
        #Calculate Error 
        self.calculate_OutputDelta(targets)
        #----------------------------------------------------------------------
        # Calculate Hidden Error
        self.calculate_HiddenDelta()
        #---------------------------------------------------------------------
    
    def TEST(self, examples, TYPE):
        #----------------------------------------------------------------------
        #Tabulate Answers
        
        predictions=[]
        actual=[]
        errors=[]

        #----------------------------------------------------------------------
        #Node count for outputting example number
        node=0
        #----------------------------------------------------------------------
        for inputObj in examples:
            inputObj=examples[0]

            #----------------------------------------------------------------------
            # Initiate Feed Foward and find predictions 
            inputNodes=inputObj[0]
            self.feed_forward( inputNodes )
            

            predictionForNode=self.output_activation

            #Return data to original size
            output_inputNodes        = inputNodes

            output_predictionForNode = predictionForNode
            #----------------------------------------------------------------------
            
            #Actual Values for data print out
            actualValues             = inputObj[1]
            output_actualValues      = actualValues

            error = self.calculateError(actualValues)
            errors.append(error)
            #------------------------------------------------------------------
            
            #Plotting Data for graph to compare predictions and output data
            predictions.append( np.array(predictionForNode) )
            
            actual.append( np.array(actualValues) )

            #------------------------------------------------------------------

            #Print Formatted Data
            print "# Test Example: {0}".format(node+1)
            print "________________________________________________"
            #for i in range(len(output_inputNodes)):
                #print "\t{0}".format(output_inputNodes[i])
            #print "________________________________________________"
            print "Output={0} Prediction={1}".format(\
                                                output_actualValues[0],\
                                                output_predictionForNode[0])
            print "________________________________________________\n"
            node+=1
            #----------------------------------------------------------------------
            #Plot Actual against predictions
            #plotThis(errors,"TestingErrors")


    ###########################################################################
    # Run Training input into backpropagation algorithm to find error 
    # calculations.
    ###########################################################################
    
    def TRAIN_WITH(self, examples, epochs, num):
    
        self.trainingID=num
        #----------------------------------------------------------------------
        # Main Execution of Training (print errors) 
        print "________________________________________________"
        
        if num == 0:
            print "\t# Training Network\n"
        else:
            print "\t# Validating Network\n"
        
        print "\tEpochs = {0}\n\tTraining Examples = {1}"\
               .format(epochs,len(examples))
        print "\tShow every {0} epochs "\
               .format(epochs/100)
        print "________________________________________________"
        print "\n(epoch number) := (error)\n"

        RMSE_=[]
        valNum=0

        for epoch in range(epochs):

            error = 0.0
            
            for obj in examples:
                #Training Input
                inputs = obj[0]          

                #Training Output *Predictions
                targets = obj[1]
                
                #Forward Pass Function                      
                self.feed_forward(inputs)             
                
                #Back Propogation Function

                self.backPropagate(targets)

                self.updateWeights(epoch)
                
                #Update Error
                error  += self.calculateError(targets)
                
            RMSE_.append(error)
            valNum=self.bold_driver(RMSE_,valNum)

            networkClone=self.cloneState()

            self.__trainingNeurons.addNeuron(epoch,networkClone,RMSE_[epoch])
            #Validation Set

            if num == 1 : self.__validationNeurons.addNeuron(epoch,networkClone,RMSE_[epoch])
            
            if valNum >= 10:
                if num == 1:
                    print "\n\t# Finished Validation\n\n"

                    selectedNeuron = ( epoch - 10 )       

                    self.__validationNeurons.setSelectedNeuron(selectedNeuron)

                    outputThisNeuron = self.__validationNeurons.getNeuron(selectedNeuron)
                    
                    show(outputThisNeuron,selectedNeuron)
                
                self.save_errors(RMSE_)
                return
            else:
                pass

            del networkClone #Delete networkClone for memory efficiency

            print "{0} = {1} LR= {2}"\
            .format(epoch,error,self.learning_rate)
     
        #----------------------------------------------------------------------
        print "\n\t# Finished Training\n\n"

        #Save Errors to File
        self.save_errors(RMSE_)
    

    ###########################################################################
    #   Save RMS Errors to File errors.txt
    ###########################################################################    
    def save_errors(self,errors):
        
        TYPE=self.trainingID

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

###########################################################################
#   Save Weights to File weights_activations.txt
########################################################################### 
    
        
    
###########################################################################
#   Generate Random Numbers for Hidden and Output Weights
###########################################################################
def generateRandFor(networkState,option):
   
    # Return Random Number ( a <= n < b )

    if option == "HI":
        a = -2.0 / networkState.number_of_inputs
        b = 2.0 / networkState.number_of_inputs
        number = (b - a) * random() + a
        return number
        
    else:
        a = -2.0 / networkState.number_of_hidden
        b = 2.0 / networkState.number_of_hidden
        number = (b - a) * random() + a
        return number



def show(neuron,epoch):

    neuronNetwork=neuron.getNetwork()
    
    fileName="Details.txt"
    weightsFile= open(fileName, "w")
    weightsFile.close()

    #Now created rewrite
    weightsFile= open(fileName, "w")

    output="\nFinal Epoch = "
    output+=str(epoch)
    output+="\nError = "
    output+=neuron.getError()

    output+="\nFinal Learning Rate: "
    output+=str(neuronNetwork.learning_rate)
    output+="\nFinal Momentum: "
    output+=str(neuronNetwork.momentum)
    output+="\n"

    output+="\n# HI Weights\n"
    output += tabulate( neuronNetwork.HI_WEIGHTS)
    
    output+="\n# HO Weights\n"
    output += tabulate( neuronNetwork.HO_WEIGHTS)
    
    output+="\n# Input Final Changes\n"
    output+= tabulate( neuronNetwork.input_change)
    
    output+="\n# Output Final Changes\n"
    output+= tabulate( neuronNetwork.output_change)
    
    output+="\n# Input Activations"
    output+="\n---------------------\n"
    for i in neuronNetwork.input_activation:
        output+=str(i)
        output+="\n"
    output+="---------------------\n"
    output+="# Output Activations"
    output+="\n---------------------\n"
    for i in neuronNetwork.output_activation:
        output+=str(i)
    output+="\n---------------------\n"
    output+="# Hidden Activations"
    output+="\n---------------------\n"
    for i in neuronNetwork.hidden_activation:
        output+=str(i)
        output+="\n"
    output+="---------------------\n"
    
    weightsFile.write(output)
    weightsFile.close()

    plotThis(neuronNetwork.input_activation,'InputActivations')
    plotThis(neuronNetwork.hidden_activation,'HiddenActivations')



#--------------------------------------------------------------------------
###########################################################################
#                       END OF NEURAL NETWORK CLASS
##########################################################################
#--------------------------------------------------------------------------
def plotThis(array,name):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #----------------------------------------------------------------------
    y1=np.array(array)
    
    x1 =np.array( vector( len(array) ) )

    
    x_smooth1 = np.linspace(x1.min(), x1.max(), 200)
    y_smooth1 = spline(x1, y1, x_smooth1)

    ax.plot(x_smooth1,y_smooth1,'r')
 

    fig.savefig(name+".pdf")  
       

###########################################################################
#   Plot graphical view of errors
###########################################################################
def plot_errors(error,TYPE):
    
    fig1 = plt.figure()
    
    fig2 = ERROR_FIGURE

    
    #----------------------------------------------------------------------
    y1=np.array(error)
    #----------------------------------------------------------------------
    

    if TYPE == 0:
        ax1 = fig1.add_subplot(1,1,1)
        ax2 = fig2.add_subplot(1,1,1)

        
        ax1.plot(y1,'r')
        fig1.savefig('TrainingSetErrors.pdf')
        ax2.plot(y1,'r')
        
    
    elif TYPE == 1:
        ax1 = fig1.add_subplot(1,1,1)
        ax2 = fig2.add_subplot(1,1,1)
        
        ax1.plot(y1,'g')
        fig1.savefig('ValidationSetErrors.pdf')
        ax2.plot(y1,'g')
       
        
       
        fig2.savefig("ErrorPlot.pdf")
        
    
    


###########################################################################
#   Take Output readings and Prediction readings and plot 
#   against each other.
###########################################################################
def smooth_plot(actual,pred,TYPE):
    
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
    
    if TYPE == 0:
        fig.savefig('Training_Predictions.pdf')

    elif TYPE == 1:
        fig.savefig('Validation_Predictions.pdf')
    


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
    
    DATA_SET        =   300
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

    #TRAINING_DATA=AND
    #VALIDATION_DATA=AND
    #TESTING_DATA=AND

    #----------------------------------------------------------------------
    # Assign lengths
    inputs          = len(TRAINING_DATA[0][0])
    hidden          = len(TRAINING_DATA[0][0])
    outputs         = len(TRAINING_DATA[0][1])
    #----------------------------------------------------------------------
    # Create MultiLayer Perceptron Network
    
    #----------------------------------------------------------------------
    # Train Network with 400 training examples (> 100 epochs) and (> 10 training patterns)

    #Start Testing data at 0 or ....
  
    #------------------------------------------------------------------
    
    Network         = NETWORK ( inputs, hidden, outputs)    
    
    #Training Set
    Network.TRAIN_WITH(  TRAINING_DATA  ,500,0)

    #Validation Set
    Network.TRAIN_WITH(  VALIDATION_DATA,200,1)

    #Testing Set
    Network.TEST(TESTING_DATA,0)

    return Network

    #------------------------------------------------------------------

network=execute_MLP()

