#   Back Propogation Algorithm (Multi-layer Perceptron) (Main file)
#   @Copywright Max Pearson
#   Student ID: B123103


from tabulate import tabulate
import os
import math
from MLP import *
from Data import *




# -------------------------------------------
#               Main Program 
# -------------------------------------------
def runProgram():
    data=Data()
    Area=data.getBy("AREA")
    
    #Start Algorithm
    mlpObject=MLP()
    mlpObject=presentExample(mlpObject,Area)
    print "-------------------------------------------"
    print "-------------------------------------------"
    print "1. Run MLP Program "
    print "2. View Data "
    print "-------------------------------------------"
    print "-------------------------------------------\n\n"
    option=raw_input("Please pick option: ")

    count=0
    if (option == '1'):
        while True:
            try:
                count=count+1
                #Program main
                mlpObject=determine_hidden(mlpObject)
                mlpObject=determine_output(mlpObject)
                mlpObject=error_output(mlpObject)
                mlpObject=error_hidden(mlpObject)

                
            except Exception as e:
                print 'error found'

            finally:
                print "pass: "

    elif (option == '2'):
        showLegend(data)
        print "\n\n"
        showData(data)

    else:
        pass
# -------------------------------------------
# -------------------------------------------

if __name__== "__main__":
    runProgram()
    
    
        
