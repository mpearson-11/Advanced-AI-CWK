#   @Copywright Max Pearson
#   Student ID: B123103
# 	Date Created 21/03/2015

###########################################################################
#                         ------------------------
#                         |     Neuron class    |
#                         ------------------------
###########################################################################


class Neuron:
    
    def __init__(self,index,Network,error):
        
        self.master		    = Network
        self.index 			= index
        self.error 		    = error
        self.selectedNeuron = False

    def getNetwork(self):
        return self.master

    def getIndex(self):
        return self.index

    def getError(self):
   		return str(self.error)

   	


###########################################################################