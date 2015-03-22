#   @Copywright Max Pearson
#   Student ID: B123103
#   Date Created 21/03/2015

###########################################################################
#                         ------------------------
#                         |     Neurons class    |
#                         ------------------------
###########################################################################
from Neuron import *

class Neurons:
    
    def __init__(self):
        #Neurons object list accessed by index or getNeuron(index)
        self.neurons=[]
        self.last=0

    #Set New Neuron with NETWORK state
    def addNeuron(self,index,currentNetworkState,error):
        self.neurons.append(Neuron(index,currentNetworkState,error))
        self.last=len(self.neurons)-1

    #Get Neuron via index
    def getNeuron(self,index):
        return self.neurons[index] #Return Neuron Class

    def getNeuronNetwork(self,index):
        return self.getNeuron(index).getNetwork() #Return Network Class

    #Return Neuron that is selected with minimum error
    def getSelectedNeuron(self):
        for neuron in self.neurons:
            isSelected=neuron.selectedNeuron
            if isSelected == True:
                return neuron
            

    #Return Neuron that is selected with minimum error
    def setSelectedNeuron(self,index):
        self.getNeuron(index).selectedNeuron=True


###########################################################################
###########################################################################

