"""coordinates
Some functions to create and read coordinates
 
So far: Creates random coordinates; reads and xyz file
"""
import numpy as np
#import quippy as qp
#import chemcoord as chc
#import ase.io

## Chemical system type 
# To be used only when really needed! 
# 
class system:
    """A prototype for the system type.
    """
    def __init__(self):
        self.nats = 1 #Number of atoms
        self.ntypes = 1 #Number of atom types
        self.types = np.zeros(self.nats,dtype=int) #Type of each atom
        self.coords = np.zeros((self.nats,3)) #Coordinates for each atom
        self.symbols = ["Bl"] * self.ntypes #Symbols for each atom type


## Transforms the lattice parameters into lattice vectors.
# @param paramA a parameter
# @param paramB b parameter
# @param paramC c parameter 
# @param angleAlpha Angle beween second and third lattice vectors
# @param angleBeta Angle between first and third lattice vectors
# @param angleGamma Angle between first and second lattice vectors
# @param latticeVectors 3x3 array containing the lattice vectors.
# latticeVector[0,2] = z-coordinate of the first lattice vector
# @param verb Verbosity level.
#
def parameters_to_vectors(paramA,paramB,paramC,angleAlpha,angleBeta,angleGamma,\
        latticeVectors,verb=False):
    """Transforms parameters to vectors"""

    pi = 3.1415926535897932384626433832795

    angleAlpha = 2.0*pi*angleAlpha/360.0
    angleBeta = 2.0*pi*angleBeta/360.0
    angleGamma = 2.0*pi*angleGamma/360.0

    latticeVectors[0,0] = paramA
    latticeVectors[0,1] = 0
    latticeVectors[0,2] = 0

    latticeVectors[1,0] = paramB*np.cos(angleGamma)
    latticeVectors[1,1] = paramB*np.sin(angleGamma)
    latticeVectors[1,2] = 0

    latticeVectors[2,0] = paramC*np.cos(angleBeta)
    latticeVectors[2,1] = paramC*( np.cos(angleAlpha) - np.cos(angleGamma)* \
         np.cos(angleBeta) )/np.sin(angleGamma)
    latticeVectors[2,2] = np.sqrt(paramC**2 - latticeVectors[2,0]**2 - latticeVectors[2,1]**2)

    return latticeVectors


class rand:
    """To generate random numbers.
    """
    def __init__(self,seed):
        self.a = 475
        self.b = 38
        self.c = 41
        self.seed = seed
        self.status = seed

    def get_rand(self,low,high):
        """Get a random real number in betwee low and high."""
        w = high - low
        place = self.a*self.status
        place = place/self.b
        rand = (place%self.c)/self.c
        place = rand*100000
        self.status = place
        rand = low + w*rand

        return(rand)

## Generating random coordinates 
# Creates a system of size length^3 with coorindates having 
# a random (-1,1) displacement from a simple cubic lattice 
# with parameter 2.0 Ang.
#
# @param lenght The total number of point in x, y, and z directions.
# @return coordinates Position for every atoms. z-coordinate of atom 1 = coords[0,2]
#
# \verbatim
# NumberOfAtoms = len(coordinates[:,0])
# \endverbatim
#
def get_random_coordinates(length):
    """Get random coordinates real number in betwee low and high."""
    nats = length**3
    coords = np.zeros((nats,3))
    latticeParam = 2.0 
    atomsCounter = -1
    myrand = rand(123)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,0] = i*latticeParam + rnd
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,1] = j*latticeParam + rnd
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,2] = k*latticeParam + rnd 
    return coords


## xyz file parser
#  Reads in an xyz file with lattice informations.
#
#     Example xyz file format as follows: 
#        
# \verbatim
#        3 
#        Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0"
#        O  0.0 0.0 0.0 
#        H  1.0 0.0 1.0
#        H -1.0 0.0 1.0 
# \endverbatim
#
# @param fileName File name of the xyz file. Example: "coords.xyz"
# @param lib If using a particular library. Default is "None"
# @param verb Verbosity. If set to True will output relevant information.
# @return latticeVectors Lattice vectors. z-coordinate of the first vector = latticeVectors[0,2] 
# @return symbols Symbol for each atom type. Symbol for first atom type = symbols[0]
# @return types Index type for each atom in the system. Type for first atom = type[0]
# @return coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
#
# @code{.unparsed}
# NumberOfAtomTypes = len(symbols)
# NumberOfAtoms = len(coordinates[:,0])
# @endcode
#
def read_xyz_file(fileName,lib="None",verb=True):
    """xyz file parser: Reads in an xyz file with lattice informations.
    """
    if(lib == "None"):
        fileIn = open(fileName,"r") 
        count = -1 
        latticeVectors = np.zeros((3,3))
        symbols = [] #Symbols for each atom type
        noBox = False
        typesIndex = -1
        for lines in fileIn:
            linesSplit = lines.split()
            if(len(linesSplit) != 0): 
                count = count + 1
                if(count == 0):
                    nats = int(linesSplit[0])
                    coords = np.zeros((nats,3))
                    types = np.zeros((nats),dtype=int)
                if(count == 1):
                    latticeKey = (linesSplit[0][0:8])
                    if((latticeKey == "Lattice=") or (latticeKey == "Lattice")):
                        linesSplit = lines.split('"')
                        if(linesSplit[0] == "Lattice"):
                            boxInfoList = linesSplit[2].split()
                        else:
                            boxInfoList = linesSplit[1].split()
                        #Reading the lattice vectors
                        latticeVectors[0,0] = float(boxInfoList[0])
                        latticeVectors[0,1] = float(boxInfoList[1])
                        latticeVectors[0,2] = float(boxInfoList[2])
                
                        latticeVectors[1,0] = float(boxInfoList[3])
                        latticeVectors[1,1] = float(boxInfoList[4])
                        latticeVectors[1,2] = float(boxInfoList[5])
                
                        latticeVectors[2,0] = float(boxInfoList[6])
                        latticeVectors[2,1] = float(boxInfoList[7])
                        latticeVectors[2,2] = float(boxInfoList[8])

                    else:
                        noBox = True
                if((count >= 2) and (count <= nats + 2)):
                    #Reading the coordinates
                    coords[count - 2,0] = float(linesSplit[1])
                    coords[count - 2,1] = float(linesSplit[2])
                    coords[count - 2,2] = float(linesSplit[3])
                    newSymbol = linesSplit[0]
                    if(not(newSymbol in symbols)):
                        symbols.append(newSymbol)
                        typesIndex = typesIndex + 1
                        types[count - 2] = typesIndex 
                    else:
                        types[count - 2] = symbols.index(newSymbol)

        if(noBox): 
            #If there is no box we create one by taking the coordinate
            #limits given by the positions of the atoms
            latticeVectors[0,0] = np.max(coords[:,0]) - np.min(coords[:,0])
            latticeVectors[1,1] = np.max(coords[:,1]) - np.min(coords[:,1])
            latticeVectors[2,2] = np.max(coords[:,2]) - np.min(coords[:,2])
    
    if(lib == "Ase"): #https://wiki.fysik.dtu.dk/ase/ase/atoms.html
        system = ase.io.read(fileName)
        coords = system.get_positions()
        symbols = [] #Symbols for each atom type
        latticeVectors = np.zeros((3,3))
        noBox = True #Ace xyz reader does not read lattice vectors (system.cell = 0)
        symbolsForEachAtom = system.get_chemical_symbols() 
        types = np.zeros(len(symbolsForEachAtom),dtype=int)
        typesIndex = -1
        count = -1
        for symb in symbolsForEachAtom:
            count = count + 1
            if (not(symb in symbols)):
                symbols.append(symb)
                typesIndex = typesIndex + 1
                types[count] = typesIndex
            else:
                types[count] = symbols.index(symb)

    if(verb):
        print("latticeVectors",latticeVectors)
        print("symbols",symbols)
        print("coords",coords)

    return latticeVectors,symbols,types,coords

## Read a pdb file 
#  Reads in an pdb file with lattice informations.
#
#     Example pdb file format as follows: 
#        
# \verbatim
#    TITLE coords.pdb
#    CRYST1   11.598   17.395   17.591  90.00  90.00  90.00 P 1           1
#    MODEL                   1
#    ATOM      1  O   MOL     1       0.000   0.805  -0.230  0.00  0.00          O
#    ATOM      2  H   MOL     1      -0.101   0.855   2.111  0.00  0.00          H
#    ATOM      3  H   MOL     1       0.827  -0.475   3.907  0.00  0.00          H
#    TER 
#    END
# \endverbatim
#
# @param fileName File name of the xyz file. Example: "coords.xyz"
# @param lib If using a particular library. Default is "None"
# @param verb Verbosity. If set to True will output relevant information.
# @return latticeVectors Lattice vectors. z-coordinate of the first vector = latticeVectors[0,2] 
# @return symbols Symbol for each atom type. Symbol for first atom type = symbols[0]
# @return types Index type for each atom in the system. Type for first atom = type[0]
# @return coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
#
# @code{.unparsed}
# NumberOfAtomTypes = len(symbols)
# NumberOfAtoms = len(coordinates[:,0])
# @endcode
#
def read_pdb_file(fileName,lib="None",verb=False):
    """Reads a pdb file"""
    print("In read_pdb_file")
    if(lib == "None"):
        fileIn = open(fileName,"r")
        count = 0
        latticeVectors = np.zeros((3,3))
        symbols = [] #Symbols for each atom type
        noBox = False
        typesList = []
        coordsxList = []
        coordsyList = []
        coordszList = []
        typesIndex = -1
        for lines in fileIn:
            linesSplit = lines.split()
            if(len(linesSplit) != 0):
                if(linesSplit[0] == "CRYST1"):
                    paramA = float(linesSplit[1])
                    paramB = float(linesSplit[2])
                    paramC = float(linesSplit[3])
                    paramAlpha = float(linesSplit[4])
                    paramBeta = float(linesSplit[5])
                    paramGamma = float(linesSplit[6])
                    latticeVectors = parameters_to_vectors(paramA,paramB,paramC,paramAlpha, \
                            paramBeta,paramGamma,latticeVectors)
                    print(latticeVectors)
                else:
                    noBox = True
                if((linesSplit[0] == "ATOM") or (linesSplit[0] == "HETATM")):
                    count = count + 1
                    newSymbol = linesSplit[2]
                    if(not(newSymbol in symbols)):
                        symbols.append(newSymbol)
                        typesIndex = typesIndex + 1
                        typesList.append(typesIndex)
                    else:
                        typesList.append(symbols.index(newSymbol))
                    coordsxList.append(float(linesSplit[5]))
                    coordsyList.append(float(linesSplit[6]))
                    coordszList.append(float(linesSplit[7]))

    coords = np.zeros((count,3))
    for i in range(count):
        coords[i,0] = coordsxList[i]
        coords[i,1] = coordsyList[i]
        coords[i,2] = coordszList[i]
    types = np.array(typesList,dtype=int)
    
    return latticeVectors,symbols,types,coords 

## Write coordinates into an xyz file
# 
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param symbols Symbols for every atom type
# @types list of types for every atom in the system. 
# 
def write_xyz_coordinates(coords,types,symbols):
    """Writes coordinates in simple xyz format
    """
    nats = len(coords[:,1])
    myFileOut = open("coords.xyz","w")
    print(nats,file=myFileOut)
    print("coords.xyz",file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print(symb,coords[i,0],coords[i,1],coords[i,2],file=myFileOut)



