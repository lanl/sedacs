"""sdc prototype parser"""

import sys

import numpy as np


## Input reader
# @brief This will be used to store and read the values for all the
# input variables used in the code.
#
class Input:
    """Simple input parser"""

    def __init__(self,
                 fileName: str,
                 verb: bool = False):
        """
        Constructor for the Input class.

        Parameters
        ----------
        fileName : str
            The name of the input file. 
        verb : bool
            Whether to print the input variables.

        Returns
        -------
        None
        """

        if verb:
            print("\nInput variables:")
        ## Keys and values read from the input file
        keyVals = self.get_all_vals(fileName)

        ## A key list for validation purposes
        validKeys = []

        ## A tag for naming files. First argument is the key, the second is
        # the default.
        self.tag = self.get_a_string("Tag=", "myRun", keyVals, validKeys, verb)
        ## Coordinates file name
        self.coordsFileName = self.get_a_string("CoordsFile=", "coords.xyz", keyVals, validKeys, verb)
        ## Coordinates file name
        self.InitGraphType = self.get_a_string("InitGraphType=", "regular", keyVals, validKeys, verb)
        ## Coordinates file name
        self.partitionType = self.get_a_string("PartitionType=", "regular", keyVals, validKeys, verb)
        ## Max degree for the grpah
        self.SpecClustNN = self.get_an_int("SpecClustNN=", 8, keyVals, validKeys, verb)
        ## Max degree for the grpah
        self.maxDeg = self.get_an_int("MaxDeg=", 100, keyVals, validKeys, verb)
        ## Number of parts to perform graph partitioning
        self.nparts = self.get_an_int("NumParts=", 1, keyVals, validKeys, verb)
        ## Radius cutoff
        self.rcut = self.get_a_real("Rcut=", 5.0, keyVals, validKeys, verb)
        ## Alpha for DM mixing. DM_new = (1-alpha)*DM_old + alpha*DM_new
        self.alpha = self.get_a_real("Alpha=", 0.2, keyVals, validKeys, verb)
        ## A threshold read from input
        self.thresh = self.get_a_real("Threshold=", 0.0, keyVals, validKeys, verb)
        ## A threshold for the graph
        self.gthresh = self.get_a_real("GraphThreshold=", 0.0, keyVals, validKeys, verb)
        ## A threshold for the initial graph
        self.gthreshinit = self.get_a_real("GraphThresholdInit=", 0.0, keyVals, validKeys, verb)
        ## A field read from input
        self.field = self.get_a_npFloatVect("Field=", np.zeros((3)), keyVals, validKeys, verb)
        ## Number of orbitals
        self.orbs = self.get_a_dict("Orbitals=", {"Bl": 1}, keyVals, validKeys, verb)
        ## Valency per atom type
        self.valency = self.get_a_dict("Valency=",{"Bl":1},keyVals,validKeys,verb)
        ## Flag to run open shell calc 
        self.UHF = self.get_a_bool("UHF=", False, keyVals, validKeys, verb=False)
        ## Total charge
        self.charge = self.get_an_int("charge=", 0, keyVals, validKeys, verb)
        ## Multiplicity
        self.mult = self.get_an_int("mult=", 1, keyVals, validKeys, verb)
        ## Electronic temperature 
        self.Tel = self.get_a_real("Tel=",0.0,keyVals,validKeys,verb)
        ## do scf on cpu or gpu 
        self.scfDevice = self.get_a_string("scfDevice=", "cpu", keyVals, validKeys, verb)
        ## Flag to do forces calculation 
        self.doForces = self.get_a_bool("doForces=", False, keyVals, validKeys, verb=False)
        ## calculate i-j pairs via vectorization (fast but memory consuming) or via loop
        self.ijMethod = self.get_a_string("ijMethod=", "Vec", keyVals, validKeys, verb)
        ## Number of graph jumps
        self.numJumps = self.get_an_int("numJumps=", 1, keyVals, validKeys, verb)
        ## Number of adaptive graph iterations
        self.numAdaptIter = self.get_an_int("NumAdaptiveIter=", 1, keyVals, validKeys, verb)
        ## Flag to save some arrays for restart purposes 
        self.restartSave = self.get_a_bool("restartSave=", False, keyVals, validKeys, verb=False)
        ## Flag to load some arrays for restart purposes 
        self.restartLoad = self.get_a_bool("restartLoad=", False, keyVals, validKeys, verb=False)
        ## When rumming on GPU, set the number of GPUs per node manually. Use in case of inhomogeneous nodes and set the number to the minimum number of GPUs on one node. Has no effect on CPU runs.
        self.numGPU = self.get_an_int("numGPU=", -1, keyVals, validKeys, verb)
        ## Flag to save core and choreHalo geomatries 
        self.writeGeom = self.get_a_bool("writeGeom=", False, keyVals, validKeys, verb=False)
        ## Engine interface type
        self.engineInterfaceType = self.get_a_string("EngineInterfaceType=", "Files", keyVals, validKeys, verb)
        ## Engine name
        self.engineName = self.get_a_string("EngineName=", "ProxyA", keyVals, validKeys, verb)
        ## Engine data
        self.engine = self.get_a_dict(
            "Engine=",
            {"Name": "ProxyA", "InterfaceType": "Module", "Path": "/tmp/",
                "Executable": "/tmp/run","RhoSolverMethod":"Diag","Accelerator":"No"},
            keyVals,
            validKeys,
            verb,
        )
        ## Verbosity switch
        self.verb = self.get_a_bool("Verbosity=", False, keyVals, validKeys, verb=True)
        ##Overlap (if set to True it will do a nonortho calculation)
        self.over = self.get_a_bool("Overlap=", True, keyVals, validKeys, verb=True)
        ## StopAt : stop at a given point
        self.stopAt = self.get_a_string("StopAt=", "", keyVals, validKeys, verb)
        ## SCF tolerance - For Self-consistent charge optimization
        self.scfTol = self.get_a_real("SCFTol=",0.0,keyVals, validKeys,verb)
        ## Electronic temperature 
        self.etemp = self.get_a_real("ElectronicTemperature=",0.0,keyVals, validKeys,verb)
        ## Chemical potential calculation type
        self.mucalctype =self.get_a_string("MuCalculationType=","None",keyVals, validKeys,verb)
        
        ## Will check to make sure there are only valid key name in the input
        err = self.validate_keys(keyVals, validKeys)
        if err:
            sys.exit(0)

    ## Check all the key names.
    # @brief Will check to make sure there are only valid key name in the input
    # @return keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    def validate_keys(self,
                      keyVals: dict,
                      validKeys: list) -> bool:
        """
        Check all the key names.

        Parameters
        ----------
        keyVals : dict
            A dictionary where values are list of characters after the key.
        validKeys : list
            A list with all the valid accumulated key names.

        Returns
        -------
        err : bool
            Whether there are only valid key names in the input.

        """
        for key in keyVals.keys():
            err = True
            for valid in validKeys:
                if key == valid:
                    err = False
            if err:
                print("\n!!!ERROR: Invalid keyword", key)
                print("\nValid keywords are the following:", validKeys)
                break

        return err

    ## Get all the values in the input
    # @brief Will return a dict with key:val, where val is a list
    # @param fileName Name of input file
    # @return keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_all_vals(self, fileName):
        keyVals = {}
        myFile = open(fileName, "r")
        for lines in myFile:
            lineList = lines.split()
            if len(lineList) != 0:
                key = lineList[0]
                firstChar = key[0]
                if firstChar != "#":  # Comment character
                    listIn = []
                    # Collect everything between comments
                    for i in range(1, len(lineList)):
                        if lineList[i] != "#":
                            listIn.append(lineList[i])
                        else:
                            break
                    keyVals.update({key: listIn})
        return keyVals

    ## Get a string
    # @brief Extracts a string value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_a_string(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            myString = keyVals[key][0]
        else:
            myString = default
        if verb:
            print("Input: ", key, myString)
        validKeys.append(key)
        return myString

    ## Get a real value
    # @brief Extracts a real value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_a_real(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            myReal = float(keyVals[key][0])
        else:
            myReal = default
        if verb:
            print("Input: ", key, myReal)
        validKeys.append(key)
        return myReal

    ## Get an integer value
    # @brief Extracts an integer value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_an_int(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            myInt = int(keyVals[key][0])
        else:
            myInt = default
        if verb:
            print("Input: ", key, myInt)
        validKeys.append(key)
        return myInt

    ## Get a boolean value
    # @brief Extracts a boolean value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_a_bool(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            if(keyVals[key][0] == "True"):
                myBool = True
            else:
                myBool = False
        else:
            myBool = default
        if verb:
            print("Input: ", key, myBool)
        validKeys.append(key)
        return myBool

    ## Get a numpy vector of type float
    # @brief Extracts a numpy vector value from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_a_npFloatVect(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            myVect = np.zeros((len(keyVals[key])))
            for i in range(len(keyVals[key])):
                myVect[i] = float(keyVals[key][i])
        else:
            myVect = default
        if verb:
            print("Input: ", key, myVect)
        validKeys.append(key)
        return myVect

    ## Get a dictionary
    # @brief Extract a dictionary from the keyVals dict
    # @param key Key to search in the dictionary
    # @param deafult Default value in case it is not in the dict
    # @param keyVals A dictionary where values are list of characters after the key
    # @param validKeys A list with all the valid accumulated key names
    #
    def get_a_dict(self, key, default, keyVals, validKeys, verb=False):
        if key in keyVals.keys():
            myDict = {}
            myDict = eval(keyVals[key][0])
        else:
            myDict = default
        if verb:
            print("Input: ", key, myDict)
        validKeys.append(key)
        return myDict


if __name__ == "__main__":
    # Initialize the input variables
    inp = Input("input.in", True)
