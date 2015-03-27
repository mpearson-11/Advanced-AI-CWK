

import math
import string
from Utility import * 

def make2D(inner, outer):
    vector = []
    for i in range(inner):
        vector.append([0.0]*outer)
    return vector

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x) )

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return ( 1.0 - y ) * y


class Weighting:
    def __init__(self,i,h,o):
        
        self.w_i=i
        self.w_h=h
        self.w_o=o

        self.inputWeight=make2D(i,h)
        self.outputWeight=make2D(h,o)

    def getInputVia(self,i,j):
        return self.inputWeight[i][j]

    def getOutputVia(self,j,k):
        return self.outputWeight[j][k]

    def generateRandom(self):
        # set them to random vaules
        for i in range(self.w_i):
            for j in range(self.w_h):
                self.inputWeight[i][j] = generateRandFor(0)
        
        for j in range(self.w_h):
            for k in range(self.w_o):
                self.outputWeight[j][k] = generateRandFor(1)



class Nodes:
    def __init__(self,n,name):
        self.size=n
        self.activation =[1.0] * self.size

        if name == "Hidden" or name == "Output":
            self.delta = [1.0] * self.size

    def getSize(self):
        return self.size

    def resetDelta(self):
        self.delta = [1.0] * self.size





class Network:
    def __init__(self, ni, nh, no):
        
        self.learning_rate=0.5
        self.momentum=0.9

        self.nm_i=ni + 1
        self.nm_h=nh
        self.nm_o=no

        self.weights=Weighting(self.nm_i,self.nm_h,self.nm_o)
        self.weights.generateRandom()
        self.changes=Weighting(self.nm_i,self.nm_h,self.nm_o)

        self.inputLayer=Nodes(self.nm_i,"Input")
        self.hiddenLayer=Nodes(self.nm_h,"Hidden")
        self.outputLayer=Nodes(self.nm_o,"Output")



    def feed_forward(self,inputs):
        # input activations
        for i in range(self.inputLayer.getSize()-1):
            self.inputLayer.activation[i] = inputs[i]

        # hidden activations
        for j in range(self.hiddenLayer.getSize()):
            sumOf = 0.0
            for i in range(self.inputLayer.getSize()):
                sumOf = sumOf + self.inputLayer.activation[i] * self.weights.getInputVia(i,j)
            self.hiddenLayer.activation[j] = sigmoid(sumOf)

        # output activations
        for k in range(self.outputLayer.getSize()):
            sumOf = 0.0
            for j in range(self.hiddenLayer.getSize()):
                sumOf = sumOf + self.hiddenLayer.activation[j] * self.weights.getOutputVia(j,k)
            self.outputLayer.activation[k] = sigmoid(sumOf)

    def backPropagate(self,targets):

        self.outputLayer.resetDelta()
        for k in range(self.outputLayer.getSize()):
            error = targets[k]-self.outputLayer.activation[k]
            self.outputLayer.delta[k] = dsigmoid(self.outputLayer.activation[k]) * error

        self.hiddenLayer.resetDelta()
        for j in range(self.hiddenLayer.getSize()):
            error = 0.0
            for k in range(self.outputLayer.getSize()):
                error = error + self.outputLayer.delta[k]*self.weights.outputWeight[j][k]
            self.hiddenLayer.delta[j] = dsigmoid(self.hiddenLayer.activation[j]) * error

    def updateWeights(self,targets):
        
        #Update Hidden Layer Weights
        for j in range(self.hiddenLayer.getSize()):
            for k in range(self.outputLayer.getSize()):
                changeJK = self.outputLayer.delta[k]*self.hiddenLayer.activation[j]

                self.weights.outputWeight[j][k]=self.weights.outputWeight[j][k]+\
                (self.learning_rate * changeJK)+\
                (self.momentum * self.changes.outputWeight[j][k])
                self.changes.outputWeight[j][k]=changeJK

        #Update Input weights
        for i in range(self.inputLayer.getSize()):
            for j in range(self.hiddenLayer.getSize()):
                change = self.hiddenLayer.delta[j]*self.inputLayer.activation[i]

                self.weights.inputWeight[i][j] = self.weights.inputWeight[i][j] + \
                (self.learning_rate * change) + \
                (self.momentum * self.changes.inputWeight[i][j])
                self.changes.inputWeight[i][j] = change


        error = self.getError(targets)
        return error
    
    def getError(self,targets):
        error=0.0
        
        for k in range(self.outputLayer.getSize()):
            error += math.pow(targets[k]-self.outputLayer.activation[k],2.0)
        
        return error / 2.0

    def test(self,patterns):
        actual=[]
        pred=[]
        for examples in patterns:
            actual.append(examples[1][0])
            self.feed_forward(examples[0])
            pred.append(self.inputLayer.activation[0])

        smooth_plot(actual,pred,"TestGraph.txt","Testing Data Output vs Predictions")


    def train(self, patterns,name):
        # N: learning rate
        # M: momentum factor
        self.learning_rate=0.5
        exitCounter=0
        epoch=0
        outputErrors=[]
        ERRORS=[]
        while True:
            error = 0.0
            for examples in patterns:

                self.feed_forward(examples[0])
                self.backPropagate(examples[1])
                error += self.updateWeights(examples[1])
            
            error=math.sqrt(error)
            ERRORS.append(error)
            outputErrors.append(error)
            exitCounter=self.bold_driver(ERRORS,exitCounter)

            if exitCounter > 10 or epoch > 10000:
                print "Epochs: "+str(epoch)
                print "Exit Counter: "+str(exitCounter)
                break
            
            else:
                if epoch % 100 == 0:
                  print "Epoch {0} Error: {1} Learning Rate: {2}".format(epoch,error,self.learning_rate)
                epoch+=1

        plot_errors(outputErrors,name)

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

def execute():


    data = Data()
    #----------------------------------------------------------------------
    # Create Training and Test Data from CWKData.xlsx
    # Take 80% of data (400 epochs as default) and leave rest for testing

    DATA_SET        =   200
    INPUTS          =   8

    #Data Variables
    T_START         =   2
    T_END           =   T_START    + int(DATA_SET * 0.8)

    TTS_START       =   T_END
    TTS_END         =   TTS_START  + int(DATA_SET * 0.10)

    TST_START       =   TTS_END
    TST_END         =   TST_START  + int(DATA_SET * 0.10)    


    #CREATE NORMALISED DATA FROM FILE
    TRAINING_DATA   = createNormalisedDataSet(  T_START,   T_END,   data,  INPUTS)
    VALIDATION_DATA = createNormalisedDataSet(  TTS_START, TTS_END, data,  INPUTS)
    TESTING_DATA    = createNormalisedDataSet(  TST_START, TST_END, data,  INPUTS)
    inputs          = len(TRAINING_DATA[0][0])
    hidden = 4
    outputs         = len(TRAINING_DATA[0][1])

    # create a network with two input, two hidden, and one output nodes
    n = Network(inputs, hidden, outputs)
    # train it with some patterns
    print "Train with Training Data"
    n.train( TRAINING_DATA ,"TrainingSet")

    print "Train with Validation Data"
    n.train( VALIDATION_DATA ,"ValidationSet")
    # test it
    n.test( TESTING_DATA )



if __name__ == '__main__':
    execute()