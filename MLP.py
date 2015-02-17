#   MLP Controller
#   @Copywright Max Pearson
#   Student ID: B123103



import os
import math


class MLP:
    def __init__(self):
        self.n_in=100 #number of input nodes
        self.n_hid=10 #number of hidden nodes
        self.n_out=2 #number of output nodes
        self.eta=0.1 #learning rate 
        self.tr_examp=20 #training examples

        self.input=[]
        self.hidden=[]
        self.output=[]
        
        self.ih_weight=[[0 for x in range(self.n_in)] for x in range(self.n_hid)] 
        self.ho_weight=[[0 for x in range(self.n_hid)] for x in range(self.n_out)] 
        self.stim= [[0 for x in range(self.tr_examp)] for x in range(self.n_in)] 
        self.resp= [[0 for x in range(self.tr_examp)] for x in range(self.n_out)] 


    def zeroVector(self,vector,n1,n2):
     
        for i in range(n1):
            for j in range(n2):
                vector[i][j]=0
        return vector

    def set_inputs(self,inputs):
        for j in range(self.n_in):
            self.input.append(inputs[j])

    def setStim(self):
        pass
    
    def setResp(self):
        pass

    def resetMLP(self,N_IN,N_HID,N_OUT,ETA,TR_EXAMP):
        self.n_in=N_IN #number of input nodes
        self.n_hid=N_HID #number of hidden nodes
        self.n_out=N_OUT #number of output nodes
        self.eta=ETA #learning rate 
        self.tr_examp=TR_EXAMP #training examples

        self.input=[]
        self.hidden=[]
        self.output=[]
        
        self.ih_weight=[[0 for x in range(self.n_in)] for x in range(self.n_hid)] 
        self.ho_weight=[[0 for x in range(self.n_hid)] for x in range(self.n_out)] 
        self.stim= [[0 for x in range(self.tr_examp)] for x in range(self.n_in)] 
        self.resp= [[0 for x in range(self.tr_examp)] for x in range(self.n_out)] 
    
 

#Squash Activation 
def squash(Activation,weight,type):
    S=[0]
    for i in range(len(weight)):
        for j in range(len(weight[0])):
            S[i]=S[i]+weight[i][j]*Activation[j]

    for i in range(len(Activation)):
        Activation[i]= 1 / ( 1 + math.pow( math.e , (-1 * s[i]) ))

    if(type=="Hidden"):
        mlp.hidden=Activation
    
    if(type=="Output"):
        mlp.output=Activation
    
    return mlp

#Update Weights
def update_weights(self,weight,error,x,y):
    Hidden=mlp.hidden
    for j in range(x):
        for k in range(y):
            weight[j][k]=weight[j][k] + mlp.eta * error[k] * Hidden[j]
    return weight

def zerosN(array,n):
    for i in range(n):
        array.append(0);
    return array



def presentExample(mlp,inputs):
    mlp.set_inputs(inputs)
    return mlp

#Determine hidden layer activation
def determine_hidden(mlp):
    
    try:
        N_Hid=mlp.n_hid
        N_In=mlp.n_in
        Input=mlp.input
        IH_WEIGHT= mlp.ih_weight
        Output=mlp.output

        for j in range(N_Hid):
            for i in range(N_In):
                mlp.hidden[j]=mlp.hidden[j]+Input[i] * IH_WEIGHT[i][j]

        mlp=squash(mlp.hidden,IH_WEIGHT,"Hidden")
    
    except Exception as e:
        print "Error found in determining hidden layer activation"
        raise 

    
    finally:
        return mlp

#Determine output layer activation
def determine_output(mlp):
    try:
        N_Hid=mlp.n_hid
        N_Out=mlp.n_out
        Hidden=mlp.hidden
        HO_WEIGHT=mlp.ho_weight

        for k in range(N_Out):
            for j in range(0, N_Hid):
                mlp.output[k]=mlp.output[k]+Hidden[j] * HO_WEIGHT[j][k]

        mlp=squash(mlp.output,HO_WEIGHT,"Output")
    
    except Exception as e:
        print "Error found in determining output layer activation"
        raise Exception
    
    finally:
        return mlp

#Determine Error for Output Layer
def error_output(mlp):

    try:
        Delta=[]
        N_Out=mlp.n_out
        N_Hid=mlp.n_hid
        Resp=mlp.resp
        Output=mlp.output
        Rand=randomGenerator()
        Delta=zerosN(Delta,N_Out)
        HO_Weight=mlp.ho_weight
        
        for k in range(N_Out):
            Delta[k]=Resp[Rand,0] - Output[k]

        mlp.ho_weight=update_weights(HO_Weight,Delta,N_Out,N_Hid)
    
    except Exception as e:
        print "Error found in determining error for output layer"
        raise Exception

    finally:
        return mlp

#Determine Error for Hidden Layer
def error_hidden(mlp):
    
    try:
        Delta=[]
        Out_Error=0
        N_Out=mlp.n_out
        N_Hid=mlp.n_hid
        Hidden=mlp.hidden
        HO_Weight=mlp.ho_weight
        IH_Weight=mlp.ih_weight
        Delta=populateArray(Delta,N_Hid)
        
        for j in range(N_Hid):
            for k in range(N_Out):
                Out_Error=Out_Error + HO_Weight[j][k]*Delta[k]
            Delta[j]=Hidden[j] * (1 - Hidden[j] ) * Out_Error

        mlp.ih_weight=update_weights(IH_Weight,Delta,N_Hid,N_In)
    
    except Exception as e:
        print "Error found in determining error for output layer"
        raise Exception
    
    finally:
        return mlp   
        
