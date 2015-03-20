#   @Copywright Max Pearson
#   Student ID: B123103
#
#   Multilayer Perceptron using Backpropagation algorithm 
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
from Data import *
#--------------------------------------------------------------------------
#Scientific Packages for plotting and array manipulation
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

#Package to neatly tabulate 2 Dimensional arrays
from tabulate import *

#--------------------------------------------------------------------------
   

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
#                         |     NETWORK class    |
#                         ------------------------
###########################################################################
class NETWORK:
    def __init__(self, number_of_inputs, number_of_hidden, \
                       number_of_outputs , maximumV ):
        
        self.epochs=0
        #------------------------------------------------------------------
        #   Neural Network Settings

        self.number_of_inputs  = number_of_inputs + 1    # 1 more for bias 

        self.number_of_hidden , self.number_of_outputs = (\
        number_of_hidden      , number_of_outputs        ) 
        
        #Default Learning Rate and Momentum (subject to change)
        self.learning_rate , self.momentum, self.maximumV = ( \
        0.50               , 0.9,            maximumV        )

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
        
        self.IH_WEIGHTS    = populateVector(self.number_of_inputs,\
                                            self.number_of_hidden)

        self.HO_WEIGHTS    = populateVector(self.number_of_hidden,\
                                            self.number_of_outputs)

        # Final Values for Change in weights for Momentum   
        self.input_change  = populateVector(self.number_of_inputs,\
                                            self.number_of_hidden)

        self.output_change = populateVector(self.number_of_hidden,\
                                            self.number_of_outputs)

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
                self.IH_WEIGHTS[i][j] = self.generateRandFor("IH")

        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                self.HO_WEIGHTS[j][k] = self.generateRandFor("HO")
        #----------------------------------------------------------------------

    ###########################################################################
    # Run FeedForward function for (testing and training) setting 
    # hidden activations and return output activations
    ###########################################################################

    def feed_forward(self, inputs):
        #----------------------------------------------------------------------
        #Input Activations (-1 for lack of bias)
        for i in range(self.number_of_inputs-1):
            self.input_activation[i] = activation_function(inputs[i])
        #----------------------------------------------------------------------
        #Hidden Activations
        for j in range(self.number_of_hidden):
            sumOfHidden = 0.0
            
            for i in range(self.number_of_inputs):
                
                sumOfHidden  = sumOfHidden  \
                + self.input_activation[i] * self.IH_WEIGHTS[i][j]
                                        
            self.hidden_activation[j] = activation_function(sumOfHidden)
        #----------------------------------------------------------------------
        #Output Activations
        for k in range(self.number_of_outputs):
            sumOfOutput = 0.0
            for j in range(self.number_of_hidden):
                
                sumOfOutput = sumOfOutput \
                + self.hidden_activation[j] * self.HO_WEIGHTS[j][k]
            
            self.output_activation[k] = activation_function(sumOfOutput)

        #----------------------------------------------------------------------
        return self.output_activation

    ###########################################################################
    # Calculate Output Errors
    ###########################################################################
    
    def calculate_output_error(self,targets,delta):
        #----------------------------------------------------------------------
        for k in range(self.number_of_outputs):
            error = targets[k] - self.output_activation[k]
            delta[k] = derivative_function(self.output_activation[k]) * error
        #----------------------------------------------------------------------
        return delta

    ###########################################################################
    # Calculate Hidden Error
    ###########################################################################
    
    def calculate_hidden_error(self,delta):
        #----------------------------------------------------------------------
        for j in range(self.number_of_hidden):
            error = 0.0
            for k in range(self.number_of_outputs):
                error = error + (delta[k] * self.HO_WEIGHTS[j][k])
            delta[j] =  derivative_function(self.hidden_activation[j]) * error
            

        #----------------------------------------------------------------------
        return delta

    def bold_driver(self,old,new):
        if new > old:
            self.learning_rate*=0.5
        else:
            self.learning_rate*=1.1

        return new
        
        
    ###########################################################################
    # Update Hiddden Output weights
    ###########################################################################
    
    def update_HO(self,deltas):
        #----------------------------------------------------------------------
       
        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                change = deltas[k] * self.hidden_activation[j]
                
                self.HO_WEIGHTS[j][k] = self.HO_WEIGHTS[j][k]\
                + (self.learning_rate  * change) \
                + (self.momentum       * self.output_change[j][k])
                
                self.output_change[j][k] = change
        #----------------------------------------------------------------------
    
    ###########################################################################
    # Update Input Hiddden weights
    ###########################################################################
    
    
    def update_IH(self,deltas):
        
        #----------------------------------------------------------------------
        
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden):
                change = deltas[j]*self.input_activation[i]
                
                
                
                self.IH_WEIGHTS[i][j] = self.IH_WEIGHTS[i][j]\
                + (self.learning_rate  * change) \
                + (self.momentum       * self.input_change[i][j])
                
                self.input_change[i][j] = change
        #----------------------------------------------------------------------
    
    ###########################################################################
    #   Back-Propogation function to update IH,HO and 
    #   Delta weights and return error calculation
    ###########################################################################
    
    def backPropagate(self, targets):
        #----------------------------------------------------------------------
        if len(targets) != self.number_of_outputs:
            print "Cannot create Outputs, targets size doesnt equal outputs"
            exit()
        #----------------------------------------------------------------------
        #Calculate Error 
        outputDELTA = [0.0] * self.number_of_outputs
        outputDELTA = self.calculate_output_error(targets,outputDELTA)
        #----------------------------------------------------------------------
        # Calculate Hidden Error
        hiddenDELTA = [0.0] * self.number_of_hidden
        hiddenDELTA = self.calculate_hidden_error(hiddenDELTA)
        #----------------------------------------------------------------------
        #Update Output Hidden Weights
        self.update_HO(outputDELTA)
        #----------------------------------------------------------------------
        #Update Input Hidden Weights
        self.update_IH(hiddenDELTA)
        #----------------------------------------------------------------------
        #Calculate Errors
        error = 0.0
        normal_error=0.0
        
        for k in range(len(targets)):
            sq_rtError=targets[k]-self.output_activation[k]


            #Root Mean Squared Error
            error += math.pow(sq_rtError,2)/ len(targets)
            normal_error += sq_rtError


        #----------------------------------------------------------------------
        
        
        
        return error
    
    ###########################################################################
    #   With Hidden and Output Wieghts set and errors found
    #   run test data through forward pass function to output predictions.
    ###########################################################################
   
    def TEST(self, examples):
        #----------------------------------------------------------------------
        #Tabulate Answers
        plot={}
        plot["Prediction"]=[]
        plot["Actual"]=[]

        
        #----------------------------------------------------------------------
        #Node count for outputting example number
        node=0
        #----------------------------------------------------------------------
        for inputObj in examples:

            #----------------------------------------------------------------------
            # Initiate Feed Foward and find predictions 
            inputNodes=inputObj[0]
            predictionForNode=self.feed_forward( inputNodes )

            #Return data to original size
            output_inputNodes        = np.array(inputNodes) \
                                            * self.maximumV

            output_predictionForNode = np.array(predictionForNode) \
                                            * self.maximumV 
            #----------------------------------------------------------------------
            
            #Actual Values for data print out
            actualValues             = inputObj[1]
            output_actualValues      = np.array(actualValues) * self.maximumV  
            #------------------------------------------------------------------
            
            #Plotting Data for graph to compare predictions and output data
            plot["Prediction"].append(\
                        np.array(predictionForNode)   * self.maximumV)
            
            plot["Actual"].append(\
                        np.array(actualValues)        * self.maximumV)

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
        smooth_plot( plot["Actual"] , plot["Prediction"], self.epochs )
        #----------------------------------------------------------------------
    

    ###########################################################################
    # Run Training input into backpropagation algorithm to find error 
    # calculations.
    ###########################################################################
    
    def TRAIN_WITH(self, examples, epochs):
 
        #----------------------------------------------------------------------
        # Main Execution of Training (print errors) 
        print "________________________________________________"
       
        print "\t# Training Network\n"
        
        print "\tEpochs = {0}\n\tTraining Examples = {1}"\
               .format(epochs,len(examples))
        print "\tShow every {0} epochs "\
               .format(epochs/100)
        print "________________________________________________"
        print "\n\t(epoch number) := (error)\n"

        RMSE_=[]
        for epoch in range(epochs):
            
            error = 0.0
            normal=0.0
            
            for obj in examples:
                #Training Input
                inputs = obj[0]                       

                #Training Output *Predictions
                targets = obj[1]
                
                #Forward Pass Function                      
                self.feed_forward(inputs)             
                
                #Back Propogation Function
                RMSE=\
                    self.backPropagate(targets)
                
                #Update Error
                error         += RMSE
                
            RMSE_.append(error)

            #Show epoch every every 1 percent...
            if epoch % (epochs/100) == 0:
                
                print "\t({0}) := ({1})"\
                .format(epoch,error)

            
        #----------------------------------------------------------------------
       
        print "\n\t# Finished Training\n\n"
        self.save_errors(RMSE_)
        
    
        
    
    ###########################################################################
    #   Generate Random Numbers for Hidden and Output Weights
    ###########################################################################
    def generateRandFor(self,option,):
       
        # Return Random Number ( a <= n < b )

        if option == "IH":
            a = -2.0 / self.number_of_inputs
            b = 2.0 / self.number_of_inputs
            number = (b - a) * random.random() + a
            return number
            
        else:
            a = -2.0 / self.number_of_hidden
            b = 2.0 / self.number_of_hidden
            number = (b - a) * random.random() + a
            return number

    ###########################################################################
    #   Save RMS Errors to File errors.txt
    ###########################################################################    
    def save_errors(self,errors):
        
        fileName="errors.txt"
        errorsFile= open(fileName, "w")
        errorsFile.close()

        #Now created rewrite
        errorsFile= open(fileName, "w")

        output=""

        output+="\n----------------------------------------\n"
        output+="# RMSE Errors"
        output+="\n----------------------------------------\n"

        for i in range(len(errors)):
            output+=("epoch: "+str(i)+" = "+str(errors[i]))
            output+="\n"

        errorsFile.write(output)

        errorsFile.close()

        plot_errors(errors)

    ###########################################################################
    #   Save Weights to File weights_activations.txt
    ###########################################################################
    def save_weights(self):
        
        fileName="weights_activations.txt"
        weightsFile= open(fileName, "w")
        weightsFile.close()

        #Now created rewrite
        weightsFile= open(fileName, "w")

        output=""
        output+="-------------------------\n"
        output+="# IH Weights"
        output+="\n-------------------------\n"
        output += tabulate( self.IH_WEIGHTS)
        
        output+="\n-------------------------\n"
        output+="# HO Weights"
        output+="\n-------------------------\n"
        output += tabulate( self.HO_WEIGHTS)
        
        
        output+="\n-------------------------\n"
        output+="# Input Final Changes"
        output+="\n-------------------------\n"
        output+= tabulate( self.input_change)
        
        
        output+="\n-------------------------\n"
        output+="# Output Final Changes"
        output+="\n-------------------------\n"
        output+= tabulate( self.output_change)

        output+="\n-------------------------\n"
        output+="# Input Activations"
        output+="\n-------------------------\n"
        

        output+="---------------------\n"
        for i in self.input_activation:
            output+=str(i)
            output+="\n"
        output+="\n---------------------\n"
        
        output+="\n-------------------------\n"
        output+="# Output Activations"
        output+="\n-------------------------\n"
        
        output+="---------------------\n"
        for i in self.output_activation:
            output+=str(i)
            output+="\n"
        output+="\n---------------------\n"

        output+="\n-------------------------\n"
        output+="# Hidden Activations"
        output+="\n-------------------------\n"
        
        output+="---------------------\n"
        for i in self.hidden_activation:
            output+=str(i)
            output+="\n"
        output+="\n---------------------\n"
        
        weightsFile.write(output)
        weightsFile.close()

        


#--------------------------------------------------------------------------
###########################################################################
#                       END OF NEURAL NETWORK CLASS
##########################################################################
#--------------------------------------------------------------------------
        
       

###########################################################################
#   Plot graphical view of errors
###########################################################################
def plot_errors(error):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    #----------------------------------------------------------------------
    y1=np.array(error)
    #----------------------------------------------------------------------
 
    ax.plot(y1,'r')

    plt.ylim((0.00,0.03))

    fig.savefig('ErrorPlots.pdf')
    fig.show()


###########################################################################
#   Take Output readings and Prediction readings and plot 
#   against each other.
###########################################################################
def smooth_plot(actual,pred,epochs):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    y1=np.array(actual)
    
    y2=np.array(pred)

    ax.plot(y1,'r')
    ax.plot(y2,'b')

    plt.title("Output (red), Prediction (blue), Epochs ="+str(epochs))
    fig.savefig('PredictionsVSActual.pdf')
    
    


###########################################################################
#   Return configured data to use in Neural Network with 
#   Training and Testing.
###########################################################################
def createConfiguredData(start,end,data,maximumV):
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
            inputNum = (data.getBy(str(j))[i]) / maximumV
            innerArray1.append(inputNum)
        #----------------------------------------------------------------------
        #Output (Actual to Compare to Prediction)
        outputNum = ( data.getBy("8")[i] ) / maximumV
        
        #Populate Inner arrays
        innerArray2.append(outputNum)
        innerArray3.append(innerArray1)
        innerArray3.append(innerArray2)
        
        pat.append(innerArray3)
    #----------------------------------------------------------------------

    return pat

###########################################################################
#                       Initiate Program 
###########################################################################
def execute_MLP(trainingAmount,testingStart):
    #----------------------------------------------------------------------
    # Call class Data (populate data structure)
    
    data = Data()
    maximumV=4800
    #----------------------------------------------------------------------
    # Create Training and Test Data from CWKData.xlsx
    # Take 80% of data (400 epochs as default) and leave rest for testing
    
    TRAINING_DATA=createConfiguredData(1,trainingAmount,data,maximumV) #TRAINING DATA

    #----------------------------------------------------------------------
    # Assign lengths
    
    inputs=len(TRAINING_DATA[0][0])
    hidden=len(TRAINING_DATA[0][0])
    outputs=len(TRAINING_DATA[0][1])
    #----------------------------------------------------------------------
    # Create MultiLayer Perceptron Network
    
    Network = NETWORK ( inputs, hidden, outputs, maximumV )    
    #----------------------------------------------------------------------
    # Train Network with 400 training examples (> 100 epochs) and (> 10 training patterns)

    #Start Testing data at 0 or ....
    start=testingStart
    while True:
        print "________________________________________________"
        option=raw_input("\nRun MLP program y/n? ")

        if option == "y":
            print "________________________________________________"
            #Epoch Input
            trainingEpochs=int(raw_input("\nHow many epochs: "))

            if(trainingEpochs < 100):
                trainingEpochs = 100

            print "________________________________________________"
            #Training samples
            trainingExamples=int(raw_input("\nNo. of test patterns: "))

            if(trainingExamples < 10):
                trainingExamples = 10

            print "________________________________________________"

            #Create Training data given input
            TESTING_DATA = createConfiguredData(start,start\
                    + trainingExamples , data , maximumV )

            #------------------------------------------------------------------
            #Initiate Neural Network Training and Testing
            Network.epochs=trainingEpochs
            
            #Train Network
            Network.TRAIN_WITH( TRAINING_DATA, trainingEpochs)
            
            #Test Network
            Network.TEST( TESTING_DATA )

            #Increment start index for next testing sample
            start+=trainingExamples

            #Save Weights to File
            Network.save_weights()
            #------------------------------------------------------------------
        
        #Exit 
        else:
            exit()

if __name__ == '__main__':
    if len(sys.argv) > 2:

        trainingAmount=int(sys.argv[1])
        testingStart=int(sys.argv[2])

        execute_MLP(trainingAmount,testingStart)
    else:
        #Default
        execute_MLP(400,400)
