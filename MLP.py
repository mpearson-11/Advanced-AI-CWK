import random
import math as Math
from Data import *
import pylab

def fill1(L1):
    return [0.0 for y in range(L1)]
            
def fill2(L1,L2):
    return [[0.0 for x in range(L2)] for y in range(L1)]

def randomGenerator(n):
    return random.randrange(0,n)

#------------------------------------------------------------ 
#----------- Multi-Layer Perceptron Learning Class ----------

class MLP:
    def __init__(self):
        self.DATA=Data()

        #Variables
        self.number_of_epochs = 100     #Number of training cycles
        self.number_of_inputs  = 3      #Number of inputs - this includes the input bias
        self.number_of_hidden  = 4      #Number of hidden units
        self.number_of_tr_patterns = 4  #Number of training patterns
        self.learning_rate_IH = 0.2     #Learning rate for input layer
        self.learning_rate_HO = 0.05    #Learning rate for output layer

        self.patternN=0.0               #Pattern Number
        self.error_this=0.0             #Pattern Number Selected to Error
        self.prediction=0.0
        self.error=0.0                  #Root-Mean-Squared Error

        #   Training DATA
        self.training_Input  = fill2(self.number_of_tr_patterns,self.number_of_inputs) 
        self.training_Output = fill1(self.number_of_tr_patterns)

        #   Outputs of Hidden Neurons
        self.hidden_neurons  = fill1(self.number_of_hidden) 

        #   Weights 
        self.IH_WEIGHTS = fill2(self.number_of_inputs,self.number_of_hidden)
        self.HO_WEGHTS = fill1(self.number_of_hidden)
        
#------------------------------------------------------------ 
#-------------- Find error in training set- -----------------

def output_hidden(mlp):
    #calculate the outputs of the hidden neurons
    #-------------------------------------- 
    for i in range (mlp.number_of_hidden):
        mlp.hidden_neurons[i] = 0.0
        #-------------------------------------- 
        for j in range (mlp.number_of_inputs):
            mlp.hidden_neurons[i] = mlp.hidden_neurons[i] + mlp.training_Input[mlp.patternN][j] * mlp.IH_WEIGHTS[j][i]
        #-------------------------------------- 
        #Tanh (hyperbolic tangent of Hidden neurons)
        mlp.hidden_neurons[i] = Math.tanh(mlp.hidden_neurons[i])
    #-------------------------------------- 

    #Calculate Output
    mlp.prediction = 0.0

    #-------------------------------------- 
    for i in range (mlp.number_of_hidden):
        mlp.prediction = mlp.prediction + mlp.hidden_neurons[i] * mlp.HO_WEGHTS[i]
    #-------------------------------------- 

    #Calculate Error
    mlp.error_this = mlp.prediction - mlp.training_Output[mlp.patternN]

#------------------------------------------------------------ 
#-------------- Calculate Changes in Hidden Output ---------- 

def change_in_output(mlp):
    for i in range (mlp.number_of_hidden):
        
        #Calculate Change in weights
        weightChange = mlp.learning_rate_HO * mlp.error_this * mlp.hidden_neurons[i]
        mlp.HO_WEGHTS[i] = mlp.HO_WEGHTS[i] - weightChange

        #Smooth
        if (mlp.HO_WEGHTS[i] < -3):
            mlp.HO_WEGHTS[i] = -3

            
        elif (mlp.HO_WEGHTS[i] > 3):
            mlp.HO_WEGHTS[i] = 3

#------------------------------------------------------------ 
#-------------- Calculate Changes in Hidden Input -----------

def change_in_hidden(mlp):
    #Adjust weights
    #-------------------------------------- 
    for i in range(mlp.number_of_hidden):
       #-------------------------------------- 
       for k in range (mlp.number_of_inputs):
           x = 1 - (mlp.hidden_neurons[i] * mlp.hidden_neurons[i])
           x = x * mlp.HO_WEGHTS[i] * mlp.error_this * mlp.learning_rate_IH
           x = x * mlp.training_Input[mlp.patternN][k]
           weightChange = x
           mlp.IH_WEIGHTS[k][i] = mlp.IH_WEIGHTS[k][i] - weightChange
       #-------------------------------------- 
    #-------------------------------------- 

#------------------------------------------------------------ 
#-------------- Initiate weights------ ---------------------- 

def  initiate_weights(mlp):
  #Initiate Weights with random number 0-5
  #-------------------------------------- 
  for i in range(mlp.number_of_hidden):
    mlp.HO_WEGHTS[i] = (randomGenerator(5) - 0.5)/2
    #-------------------------------------- 
    for j in range(mlp.number_of_inputs):
        mlp.IH_WEIGHTS[j][i] = (randomGenerator(5)  - 0.5)/5
    #-------------------------------------- 
  #-------------------------------------- 

#------------------------------------------------------------ 
#-------------- Initiate giden data ------------------------- 

def initiate_test_data(mlp):
    data=mlp.DATA
    #-------------------------------------- 
    for i in range(mlp.number_of_tr_patterns):
        #-------------------------------------- 
        for j in range((mlp.number_of_inputs-1)):
            mlp.training_Input[i][j]=data.getBy( str(j) )[i]
        #-------------------------------------- 
        #bias
        mlp.training_Input[i][mlp.number_of_inputs-1]=1.0
        mlp.training_Output[i]=data.getBy("8")[i]
    #-------------------------------------- 
        
                         
#------------------------------------------------------------ 
#-------------- Initiate training data ----------------------                   
            
#Initialise test DATA Set
def initiate_data(mlp):
    print "\n\n---Initialisation---\n"

    #   Pattern 0
    mlp.training_Input[0][0]  = 1
    mlp.training_Input[0][1]  = -1

    #   Pattern 1
    mlp.training_Input[1][0]  = -1
    mlp.training_Input[1][1]  = 1

    #   Pattern 2
    mlp.training_Input[2][0]  = 1
    mlp.training_Input[2][1]  = 1

    #   Pattern 3
    mlp.training_Input[3][0]  = -1
    mlp.training_Input[3][1]  = -1

    #   Biases
    mlp.training_Input[0][2]  = 1
    mlp.training_Input[1][2]  = 1 
    mlp.training_Input[2][2]  = 1
    mlp.training_Input[3][2]  = 1


    #Real Output for Comparison
    mlp.training_Output[0] = 1
    mlp.training_Output[1] = 1
    mlp.training_Output[2] = -1
    
#------------------------------------------------------------ 
#------------------ Show Results ----------------------------
        
def show_results(mlp):
     predictions=[]
     #-------------------------------------- 
     for patternN in range(mlp.number_of_tr_patterns):
        output_hidden(mlp)
        print "Pattern Number: {0} Actual Value:{1} Neural Net Prediction: {2}"\
        .format((patternN+1),mlp.training_Output[patternN],mlp.prediction)
     #-------------------------------------- 
     pylab.plot(mlp.training_Output)
     pylab.show()

#------------------------------------------------------------ 
#---------------- Calculate Error ---------------------------

def calculate_error(mlp):
     mlp.error = 0.0;
     
     #-------------------------------------- 
     for i in range (   mlp.number_of_tr_patterns   ):
        
        mlp.patNum = i
        
        output_hidden(mlp)
        
        mlp.error = mlp.error + (   mlp.error_this * mlp.error_this )

     #--------------------------------------        
     mlp.error = mlp.error / mlp.number_of_tr_patterns
     mlp.error = Math.sqrt( mlp.error )

#------------------------------------------------------------ 
#---------------------Run Program----------------------------
def runProgram(mlp):

    try:
        epochs=int(raw_input("how many epochs: "))
        mlp.number_of_epochs=epochs
        
        initiate_weights( mlp )
        
        initiate_data( mlp ) 
        #-------------------------------------- 
        for i in range (mlp.number_of_epochs):
            #-------------------------------------- 
            for j in range (mlp.number_of_tr_patterns):
                mlp.patternN = randomGenerator(mlp.number_of_tr_patterns)
        
                output_hidden(mlp)

                change_in_output(mlp)

                change_in_hidden(mlp)

                calculate_error(mlp)
            #-------------------------------------- 
            print "|| node : {0} || error : {1} || ".format(i,mlp.error)
        #-------------------------------------- 
        show_results(mlp)
                   
    except Exception,err:
        print err
        exit()

#------------------------------------------------------------ 
#---------------------  MAIN  -------------------------------

if __name__== "__main__":
    #Create Multi-Layer Perceptron Class
    mlp=MLP()

    #Initiate Program Loop in MAIN
    while(True):
        
        print "-------------------------------------------"
        print "-------------------------------------------"
        print "1. Run Program"
        print "2. Exit"
        print "-------------------------------------------"
        print "-------------------------------------------"
        print "-------------------------------------------"

        try:
            execute=int(raw_input("Option: "))
            if(execute==1):
                runProgram(mlp)

            #Exit on anything  not 1!!!
            else:
                exit()
                
        #Exit on Exception (Type Error)
        except Exception,err:
            print "Error"
            exit()
#------------------------------------------------------------ 
#------------------------------------------------------------
    
