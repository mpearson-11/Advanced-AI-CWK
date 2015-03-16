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
#--------------------------------------------------------------------------
#Data Class for Neural Network use
from Data import *
#--------------------------------------------------------------------------
#Scientific Packages for plotting and array manipulation
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
#--------------------------------------------------------------------------

###########################################################################
# Populate an array with indexes (graphing)
###########################################################################

def vector(n):
    vector = []
    for i in range(n):
        vector.append(i)
    return vector
#--------------------------------------------------------------------------

###########################################################################
# Populate 2d array with length outerNum and inner Length innerNum
###########################################################################

def populateVector(outerNum, innerNum):
    vector = []
    for i in range(outerNum):
        vector.append([0.0]*innerNum)
    return vector
#--------------------------------------------------------------------------
###########################################################################
# Return hyperbolic tangent
########################################################################### 

def activation_function(n):
    #net= -1 * n
    #return 1 / ( 1 + math.exp(net) )
    return math.tanh(n)
#--------------------------------------------------------------------------

###########################################################################
#                         ------------------------
#                         | Neural Network Class |
#                         ------------------------
#                         @param maximumV
#                         @param number_of_hidden
#                         @param number_of_inputs
#                         @param number_of_outputs
#                         
#
###########################################################################

#Initiate Random Generator
random.seed(0)

class NETWORK:
    def __init__(self, number_of_inputs, number_of_hidden, \
                       number_of_outputs , maximumV ):
        
        #------------------------------------------------------------------
        #   Neural Network Settings

        self.number_of_inputs  = number_of_inputs + 1    # 1 more for bias 

        self.number_of_hidden , self.number_of_outputs = (\
        number_of_hidden      , number_of_outputs        ) 
        
        #Default Learning Rate and Momentum (subject to change)
        self.learning_rate , self.momentum, self.maximumV = ( \
        0.90               , 0.40         , maximumV        )

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
        print "\t\tNeural Network\n"
        print "________________________________________________"
        print "Hidden = {0}\nOutputs = {1}\nInputs = {2} + 1 bias"\
        .format(self.number_of_hidden,\
                    self.number_of_outputs, \
                    self.number_of_inputs-1)

        print "Learning Rate = {0}\nMomentum = {1}"\
        .format(    self.learning_rate, \
                    self.momentum           )        
        #----------------------------------------------------------------------
        # Random Weight Assigment
        print "________________________________________________"
        print "# Initialising Hidden Weights\n"
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden):
                self.IH_WEIGHTS[i][j] = self.generateRandFor("IH",i,j)

        print "________________________________________________"
        print "# Initialising Output Weights\n"
        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                self.HO_WEIGHTS[j][k] = self.generateRandFor("HO",j,k)
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
            delta[k] = (1.0 - math.pow(self.output_activation[k],2)) * error
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
                error = error + delta[k] * self.HO_WEIGHTS[j][k]
            delta[j] =  (1.0 - math.pow(self.hidden_activation[j],2) ) * error
        #----------------------------------------------------------------------
        return delta

    ###########################################################################
    # Update Hiddden Output weights
    ###########################################################################
    
    def update_HO(self,deltas):
        #----------------------------------------------------------------------
        for j in range(self.number_of_hidden):
            for k in range(self.number_of_outputs):
                change = deltas[k] * self.hidden_activation[j]
                
                self.HO_WEIGHTS[j][k] = self.HO_WEIGHTS[j][k]\
                + self.learning_rate  * change \
                + self.momentum       * self.output_change[j][k]
                
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
                + self.learning_rate  * change \
                + self.momentum       * self.input_change[i][j]
                
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
        outer_error = 0.0
        for k in range(len(targets)):
            sq_rtError=targets[k]-self.output_activation[k]

            #Root Mean Squared Error
            error += (  math.pow(sq_rtError,2) / 2 )
            outer_error += sq_rtError

        #----------------------------------------------------------------------
        return (error,outer_error)

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
            #----------------------------------------------------------------------
            
            #Plotting Data for graph to compare predictions and output data
            plot["Prediction"].append(\
                        np.array(predictionForNode)   * self.maximumV)
            
            plot["Actual"].append(\
                        np.array(actualValues)        * self.maximumV)

            #----------------------------------------------------------------------

            #Print Formatted Data
            print "________________________________________________"
            print "\t# Test Example: {0}".format(node+1)
            print "[",
            for i in range(len(output_inputNodes)):
                print "{0}".format(output_inputNodes[i]),
            print "] \nOutput={0} Prediction={1}".format(\
                                                output_actualValues[0],\
                                                output_predictionForNode[0])
           
            node+=1
        #----------------------------------------------------------------------
        #Plot Actual against predictions
        smooth_plot( plot["Actual"] , plot["Prediction"] )
        #----------------------------------------------------------------------
    

    ###########################################################################
    # Run Training input into backpropagation algorithm to find error 
    # calculations.
    ###########################################################################
    
    def TRAIN_WITH(self, examples, epochs):
 
        #----------------------------------------------------------------------
        # Main Execution of Training (print errors) 
        print "________________________________________________"
        print "\n------------------------------------------------"
        print "# Training Network"
        print "------------------------------------------------"
        print "Epochs = {0}\nTraining Examples = {1}"\
               .format(epochs,len(examples))
        print "Show every {0} epochs"\
               .format(epochs/100)
        print "________________________________________________"

        errors=[]
        for epoch in range(epochs):
            error = 0.0
            outer_error = 0.0
            for obj in examples:
                #Training Input
                inputs = obj[0]                       

                #Training Output *Predictions
                targets = obj[1]
                
                #Forward Pass Function                      
                self.feed_forward(inputs)             
                
                #Back Propogation Function
                (backPropagationValue,real)=\
                    self.backPropagate(targets)       
                
                #Update Error
                error += backPropagationValue  
                outer_error += real

            #Show epoch every every 1 percent...
            if epoch % (epochs/100) == 0:
                errors.append(error)
                print "epoch:{0} > [ Error = {1} ]"\
                .format(epoch,error)
                #print " [ Epoch:{0}\tError = {1}\tReal Error = {2} ]"\
                #.format(epoch,error,real) 
        #----------------------------------------------------------------------
        print "\n------------------------------------------------"
        print "# Finished Training"
        print "------------------------------------------------"
        #plot_errors(errors)

    def generateRandFor(self,option,index1,index2):
        # Return Random Number ( a <= n < b )

        if option == "IH":
            a = -2.0 / self.number_of_inputs
            b = 2.0 / self.number_of_inputs
            number = (b - a) * random.random() + a
            print "IH_Weight[{0}][{1}] = {2}".format(index1,index2,number)
            return number
            
        else:
            a = -2.0 / self.number_of_outputs
            b = 2.0 / self.number_of_outputs
            number = (b - a) * random.random() + a
            print "HO_Weight[{0}][{1}] = {2}".format(index1,index2,number)
            return number

        
        
       

###########################################################################
#   Plot graphical view of errors
###########################################################################

def plot_errors(error):
    #----------------------------------------------------------------------
    y1=np.array(error)
    x1=np.array( vector( len(error) ) )
    x_smooth1=np.linspace(x1.min(),x1.max(),100)
    y_smooth1=spline(x1,y1,x_smooth1)
    #----------------------------------------------------------------------
    plt.title("Errors")
    p2,=plt.plot(x_smooth1,y_smooth1,'r') 
    plt.ylabel('Error Value')
    #----------------------------------------------------------------------
    plt.show() 

###########################################################################
#   Take Output readings and Prediction readings and plot 
#   against each other.
###########################################################################
def smooth_plot(actual,pred):
    
    #----------------------------------------------------------------------
    y1=np.array(actual)
    x1=np.array( vector( len(actual) ) )
    x_smooth1=np.linspace(x1.min(),x1.max(),100)
    y_smooth1=spline(x1,y1,x_smooth1)
    #----------------------------------------------------------------------
    
    y2=np.array(pred)
    x2=np.array( vector( len(pred) ) )
    x_smooth2=np.linspace(x2.min(),x2.max(),100)
    y_smooth2=spline(x2,y2,x_smooth2)
    #----------------------------------------------------------------------
    # Plotting Actual Test Data (RED)
    plt.subplot(5,1,1)
    plt.title("Actual")
    p2,=plt.plot(x_smooth1,y_smooth1,'r')
    plt.ylabel('Actual Reading')  
    #----------------------------------------------------------------------
    # Plotting Prediction Test Data (BLUE)
    plt.subplot(5,1,3)
    plt.title("Prediction")
    p1,=plt.plot(x_smooth2,y_smooth2,'b') 
    plt.ylabel('Prediction Reading')  
    #----------------------------------------------------------------------
    plt.subplot(5,1,5)
    plt.title("Comparison")
    p3,=plt.plot(x_smooth2,y_smooth2,'b')
    p4,=plt.plot(x_smooth1,y_smooth1,'r')
    #----------------------------------------------------------------------
    plt.show()


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

        #innerArray1.append(1)
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
def execute_MLP():
    #----------------------------------------------------------------------
    # Call class Data (populate data structure)
    
    data = Data()
    maximumV=4800
    #----------------------------------------------------------------------
    # Create Training and Test Data from CWKData.xlsx
    
    TRAINING_DATA=createConfiguredData(3,83,data,maximumV) #TRAINING DATA
    TESTING_DATA=createConfiguredData(83,103,data,maximumV) #TEST DATA
    
    #----------------------------------------------------------------------
    # Assign lengths
    
    inputs=len(TRAINING_DATA[0][0])
    hidden=len(TRAINING_DATA[0][0])
    outputs=len(TRAINING_DATA[0][1])
    #----------------------------------------------------------------------
    # Create MultiLayer Perceptron Network
    
    Network = NETWORK ( inputs, hidden, outputs, maximumV )    

    Network.learning_rate=0.9
    Network.momentum=0.5
    #----------------------------------------------------------------------
    # Train Network with 2000 epochs and split into percentage
    
    Network.TRAIN_WITH( TRAINING_DATA, 4000)
    Network.TEST( TESTING_DATA )


if __name__ == '__main__':
    execute_MLP()