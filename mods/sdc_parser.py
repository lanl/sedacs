#!/usr/bin/env python3
""" sdc prototype parser 

"""
import numpy as np

## Input reader 
# @brief This will be used to store and read the values for all the 
# input variables used in the code. 
#
class sdc_input:
    """Simple input parser  
    """
    def __init__(self,fileName,verb=False):
        if(verb):print("\nInput variables:")
        ## Keys and values read from the input file
        keyVals = self.get_all_vals(fileName)

        ## A tag for naming files. First argument is the key, the second is 
        # the default.
        self.tag = self.get_a_string("Tag=","myRun",keyVals,verb)
        ## Coordinates file name
        self.coordsFileName = self.get_a_string("CoordsFile=","coords.xyz",keyVals,verb)
        ## Coordinates file name
        self.partitionType = self.get_a_string("PartitionType=","regular",keyVals,verb)
        ## Max degree for the grpah
        self.maxDeg = self.get_an_int("MaxDeg=",100,keyVals,verb)
        ## Number of parts to perform graph partitioning
        self.nparts = self.get_an_int("NumParts=",1,keyVals,verb)
        ## Radius cutoff
        self.rcut = self.get_a_real("Rcut=",5.0,keyVals,verb)
        ## A threshold read from input 
        self.thresh = self.get_a_real("Threshold=",0.0,keyVals,verb)
        ## A threshold for the graph 
        self.gthresh = self.get_a_real("GraphThreshold=",0.0,keyVals,verb)
        ## A field read from input 
        self.field = self.get_a_npFloatVect("Field=",np.zeros((3)),keyVals,verb)
        ## Number of orbitals 
        self.orbs = self.get_a_dict("Orbitals=",{"Bl":1},keyVals,verb)
        ## Number of adaptive graph iterations
        self.numAdaptIter = self.get_an_int("NumAdaptiveIter=",1,keyVals,verb)

    ## Get all the values in the input
    # @brief Will return a dict with key:val, where val is a list
    # @param fileName Name of input file
    # @return keyVals A dictionary where values are list of characters after the key
    #
    def get_all_vals(self,fileName):
        keyVals = {}
        myFile = open(fileName,"r")
        for lines in myFile:
            lineList = lines.split()
            if(len(lineList) != 0):
                key = lineList[0]
                firstChar = key[0]
                if(firstChar != "#"): #Comment character
                    listIn = []
                    #Collect everything between comments
                    for i in range(1,len(lineList)):
                        if(lineList[i] != "#"):
                            listIn.append(lineList[i])
                        else:
                            break
                    keyVals.update({key:listIn})
        return keyVals

    ## Get a string 
    # @brief Extracts a string value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    #    
    def get_a_string(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myString = keyVals[key][0]
        else:
            myString = default
        if(verb): print("Input: ",key,myString)
        return myString

    ## Get a real value
    # @brief Extracts a real value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # 
    def get_a_real(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myReal = float(keyVals[key][0])
        else:
            myReal = default
        if(verb): print("Input: ",key,myReal)
        return myReal
        
    ## Get an integer value
    # @brief Extracts an integer value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # 
    def get_an_int(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myInt = int(keyVals[key][0])
        else:
            myInt = default
        if(verb): print("Input: ",key,myInt)
        return myInt
    
    ## Get a boolean value
    # @brief Extracts a boolean value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # 
    def get_an_bool(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myBool = bool(keyVals[key][0])
        else:
            myBool = default 
        if(verb): print("Input: ",key,myBool)
        return myBool
        
    ## Get a numpy vector of type float
    # @brief Extracts a numpy vector value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # 
    def get_a_npFloatVect(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myVect = np.zeros((len(keyVals[key])))
            for i in range(len(keyVals[key])):
                myVect[i] = float(keyVals[key][i])
        else:
            myVect = default
        if(verb): print("Input: ",key,myVect)
        return myVect

    ## Get a dictionary 
    # @brief Extract a dictionary from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    #
    def get_a_dict(self,key,default,keyVals,verb=False):
        if(key in keyVals.keys()):
            myDict = {}
            myDict = eval(keyVals[key][0])
        else:
            myDict = default
        if(verb): print("Input: ",key,myDict)
        return myDict



if(__name__ == '__main__'):
    #Initialize the input variables 
    inp = sdc_input("input.in",True)





