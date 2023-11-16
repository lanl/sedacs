"""system
Some functions to create and read coordinates of a chemical system
 
So far: Creates random coordinates; reads and writes xyz and pdb files;
creates neighbor list.
"""
import numpy as np
global aseLib
try: import ase1.io; aseLib = True
except: aseLib = False
#from sdc_out import *
try:
    from mpi4py import MPI
    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool
if(mpiLib): 
    from sdc_mpi import * 

## Chemical system type 
# @brief To be used only when really needed! 
# 
class system:
    """A prototype for the system type.
    """
    def __init__(self,nats):
        ## Number of atoms
        self.nats = nats
        ## Number of core atoms 
        self.ncores = self.nats 
        ## Number of atom types 
        self.ntypes = nats
        ## Type for each atom, e.g., the first atom is of type "types[0]"
        self.types = np.zeros(self.nats,dtype=int) 
        ## Coordinates for each atom, e.g., z-coordinate of the frist atom is coords[0,2]
        self.coords = np.zeros((self.nats,3)) 
        ## Symbols for each atom type, e.g, the element symbol of the first atom is symbols[types[0]] 
        self.symbols = ["Bl"] * self.ntypes 


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

## Simple random number generator
# This is important in order to compare across codes 
# written in different languages.
#
# To initialize: 
# \verbatim
#   myRand = rand(123)
# \endverbatim
# where the argument of rand is the seed. 
#
# To get a random number between "low" and "high":
# \verbatim 
#   rnd = myRand.get_rand(low,high)
# \endverbatim
#
class rand:
    """To generate random numbers.
    """
    def __init__(self,seed):
        self.a = 321
        self.b = 231
        self.c = 13
        self.seed = seed
        self.status = seed*1000

    def get_rand(self,low,high):
        """Get a random real number in between low and high."""
        w = high - low
        place = self.a*self.status
        place = int(place/self.b)
        rand = (place%self.c)/self.c
        place = int(rand*1000000)
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

def read_coords_file(fileName,lib="None",verb=True):
    """coords file main parser: Reads in an xyz/pdb file with lattice informations.
    """
    ext = fileName[len(fileName)-3:len(fileName)]
    if(ext == "xyz"):
        latticeVectors,symbols,types,coords = \
            read_xyz_file(fileName,lib=lib,verb=False)
    elif(ext == "pdb"):
        latticeVectors,symbols,types,coords = \
            read_pdb_file(fileName,lib=lib,verb=False)
    else:
        msg = ext+" not recognized .."
        raise_error("read_coords_file",msg)

    return latticeVectors,symbols,types,coords


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
        if(aseLib == False):
            print("\n ERROR: Consider installing ASE library (https://wiki.fysik.dtu.dk/ase/ase/atoms.html) \n")
            exit(0)
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
    if(verb):print("\nIn read_pdb_file...\n")
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
                else:
                    noBox = True
                if((linesSplit[0] == "ATOM") or (linesSplit[0] == "HETATM")):
                    count = count + 1
                    if(len(linesSplit) == 11):
                        newSymbol = linesSplit[10] 
                    else:
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

## Write coordinates into a pdb file
# 
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param symbols Symbols for every atom type
# @types list of types for every atom in the system. 
# 
def write_pdb_coordinates(fileName,coords,types,symbols,molIds=np.zeros((0),dtype=int)):
    """Writes coordinates in simple pdb format
    """
    nats = len(coords[:,1])
    
    if(len(molIds) == 0):
        molIds = np.zeros((nats),dtype=int)
        molIds[:] = 1

    myFileOut = open(fileName,"w")
    print("TITLE ",fileName,file=myFileOut)
    print("CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1 ",file=myFileOut)
    print("MODEL",file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print("ATOM",'{:6d}'.format(i+1)," "+symb,"  MOL",'{:5d}'.format(molIds[i]),"    ", '{:05.3f}'.format(coords[i,0]),"", '{:05.3f}'.format(coords[i,1]), \
               "",'{:05.3f}'.format(coords[i,2])," 0.00  0.00          ",symb,file=myFileOut)
    print("TER",file=myFileOut)
    print("END",file=myFileOut)


## Write coordinates into an xyz file
# 
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param symbols Symbols for every atom type
# @types list of types for every atom in the system. 
# 
def write_xyz_coordinates(coords,types,symbols):
    """Writes coordinates in simple pdb format
    """
    nats = len(coords[:,1])
    myFileOut = open("coords.xyz","w")
    print(nats,file=myFileOut)
    print("coords.xyz",file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print(symb,coords[i,0],coords[i,1],coords[i,2],file=myFileOut)



## Extract subsystem 
def extract_subsystem(coords,types,symbols,part):
    subSyNats = len(part)
    subSyCoords = np.zeros((subSyNats,3))
    subSyTypes = np.zeros((subSyNats),dtype=int)
    for k in range(subSyNats):
        i = part[k]
        subSyCoords[k,:] = coords[i,:]
        subSyTypes[k] = types[i]
    return subSyCoords, subSyTypes

## Gets the volume of the simulation box
# @brief Given an array of lattice vectors, it return the box volume
# @param latticeVector Lattice vectors in an array. latice_vectors[0,2] means the z-coordinate
# of the first lattice vector.
# @return volBox Volume of the cell.
#
def get_volBox(latticeVectors,verb=False):
    
    volBox=0.0

    pi = 3.14159265358979323846264338327950
    a1xa2 = np.zeros((3))
    a2xa3 = np.zeros((3))
    a3xa1 = np.zeros((3))

    a1xa2[0] =  latticeVectors[0,1]*latticeVectors[1,2] - latticeVectors[0,2]*latticeVectors[1,1]
    a1xa2[1] = -latticeVectors[0,0]*latticeVectors[1,2] + latticeVectors[0,2]*latticeVectors[1,0]
    a1xa2[2] =  latticeVectors[0,0]*latticeVectors[1,1] - latticeVectors[0,1]*latticeVectors[1,0]

    a2xa3[0] =  latticeVectors[1,1]*latticeVectors[2,2] - latticeVectors[1,2]*latticeVectors[2,1]
    a2xa3[1] = -latticeVectors[1,0]*latticeVectors[2,2] + latticeVectors[1,2]*latticeVectors[2,0]
    a2xa3[2] =  latticeVectors[1,0]*latticeVectors[2,1] - latticeVectors[1,1]*latticeVectors[2,0]

    a3xa1[0] =  latticeVectors[2,1]*latticeVectors[0,2] - latticeVectors[2,2]*latticeVectors[0,1]
    a3xa1[1] = -latticeVectors[2,0]*latticeVectors[0,2] + latticeVectors[2,2]*latticeVectors[0,0]
    a3xa1[2] =  latticeVectors[2,0]*latticeVectors[0,1] - latticeVectors[2,1]*latticeVectors[0,0]

    #Get the volume of the cell
    volBox = latticeVectors[0,0]*a2xa3[0]+ latticeVectors[0,1]*a2xa3[1]+latticeVectors[0,2]*a2xa3[2]

    return volBox


## Neighbor list 
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
def build_nlist(coords,latticeVectors,rcut,rank=0,numranks=1,verb=False):

    if(verb): print("Building neighbor list ...")
    
    mpiON = False
    if(mpiLib and (numranks > 1)): mpiON = True 

    nats = len(coords[:,0])
    if(mpiON): comm = MPI.COMM_WORLD
    natsPerRank = int(nats/numranks)
    if(rank == numranks - 1):
        natsInRank = nats - natsPerRank*(numranks - 1)
    else:
        natsInRank = natsPerRank
    natsInBuff = max(natsInRank,nats - natsPerRank*(numranks - 1))

    #We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    #A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors,verb=False)
    density = 1.0
    maxneigh = int(3.14592 * (4.0/3.0) * density * rcut**3)

    #We assume the box is orthogonal
    nx = 1 + int(latticeVectors[0,0]/(2.0*rcut))
    ny = 1 + int(latticeVectors[1,1]/(2.0*rcut))
    nz = 1 + int(latticeVectors[2,2]/(2.0*rcut))
    nBox = nx*ny*nz
    maxInBox = int(density*(2.0*rcut)**3) #Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox,maxInBox),dtype=int)
    inbox[:,:] = -1
    totPerBox = np.zeros((nBox),dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats),dtype=int)
    xBox = np.zeros((nBox),dtype=int)
    yBox = np.zeros((nBox),dtype=int)
    zBox = np.zeros((nBox),dtype=int)
    ithFromXYZ = np.zeros((nx,ny,nz),dtype=int)

    minx = np.min(coords[:,0])
    miny = np.min(coords[:,1])
    minz = np.min(coords[:,2])

    smallReal = 0.0
    #Search for the box coordinate and index of every atom

    for i in range(nats):
        #Index every atom respect to the discretized position on the simulation box.
        #tranlation = coords[i,:] - origin !For the general case we need to make sure coords are > 0
        ix =  int((coords[i,0] - minx + smallReal)/(2.0*rcut)) #small box x-index of atom i
        iy =  int((coords[i,1] - miny + smallReal)/(2.0*rcut)) #small box y-index 
        iz =  int((coords[i,2] - minz + smallReal)/(2.0*rcut)) #small box z-index

        if(ix > nx or ix < 0): print("Error in box index"); exit(0)
        if(iy > ny or iy < 0): print("Error in box index"); exit(0) 
        if(iz > nz or iz < 0): print("Error in box index"); exit(0) 

        ith =  ix + iy*nx + iz*nx*ny  #Get small box index
        boxOfI[i] = ith

        #From index to box coordinates
        xBox[ith] = ix
        yBox[ith] = iy
        zBox[ith] = iz

        #From box coordinates to index  
        ithFromXYZ[ix,iy,iz] = ith

        totPerBox[ith] = totPerBox[ith] + 1 #How many per box
        if(totPerBox[ith] >= maxInBox): 
            print("Exceeding the max in box allowed")
            exit(0)
        inbox[ith,totPerBox[ith]] = i #Who is in both ith
    print(inbox[boxOfI[0],:],totPerBox[0],boxOfI[0],0)

    #For each atom we will look around to see who are its neighbors

    def get_neighs_of(i,boxOfI,ithFromXYZ,inbox,latticeVectors):    
        nlVect = np.zeros((maxneigh),dtype=int)
        nlTrVectX = np.zeros((maxneigh),dtype=int)
        nlTrVectY = np.zeros((maxneigh),dtype=int)
        nlTrVectZ = np.zeros((maxneigh),dtype=int)
        #print("atom",i)
        cnt = -1
        #Which box it beongs to
        ibox = boxOfI[i]
        #Look inside the box and the neighboring boxes
        for ix in range(-1,2):
            for iy in range(-1,2):
                for iz in range(-1,2):
                    #Get neigh box coordinate
                    jxBox = xBox[ibox] + ix
                    jyBox = yBox[ibox] + iy
                    jzBox = zBox[ibox] + iz
                    tx = 0.0 ; ty = 0.0 ; tz = 0.0
                    if(jxBox < 0):
                        jxBox = nx-1
                        tx = -1
                    elif(jxBox == nx):
                        jxBox = 0
                        tx = 1
                    if(jyBox < 0):
                        jyBox = ny-1
                        ty = -1
                    elif(jyBox == ny):
                        jyBox = 0
                        ty = 1
                    if(jzBox < 0):
                        jzBox = nz-1
                        tz = -1
                    elif(jzBox == nz):
                        jzBox = 0
                        tz = 1
                    
                    #Get the neigh box index
                    jbox = ithFromXYZ[jxBox,jyBox,jzBox]
                    #Now loop over the atoms in the jbox
                    for j in range(totPerBox[jbox]):
                        jj = inbox[jbox,j] #Get atoms in box j
                        translation = tx*latticeVectors[0,:] + ty*latticeVectors[1,:] + tz*latticeVectors[2,:]
                        coordsNeigh = coords[jj,:] + translation
                        distance = np.linalg.norm(coords[i,:] - coordsNeigh)
                        if ((distance < rcut) and (distance > 1.0E-12)):
                        #if (True == True):
                            cnt = cnt + 1
                            nlVect[cnt] = jj # jj is a neighbor of i by some translation
                            nlTrVectX[cnt] = tx
                            nlTrVectY[cnt] = ty
                            nlTrVectZ[cnt] = tz
        nlVect[0] = cnt

        return(nlVect,nlTrVectX,nlTrVectY,nlTrVectZ)

    nlChunk = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkX = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkY = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkZ = np.empty([natsInRank,maxneigh],dtype=int)

    for k in range(natsInRank):
        i = natsPerRank*(rank) + k 
        nlVect,nlTrVectX,nlTrVectY,nlTrVectZ = get_neighs_of(i,boxOfI,ithFromXYZ,inbox,latticeVectors)
        nlChunk[k,:] = nlVect[:] 
        nlTrChunkX[k,:] =  nlTrVectX[:]
        nlTrChunkY[k,:] =  nlTrVectY[:]
        nlTrChunkZ[k,:] =  nlTrVectZ[:]

    nl = np.empty([nats,maxneigh],dtype=int)
    nlTrX = np.empty([nats,maxneigh],dtype=int)
    nlTrY = np.empty([nats,maxneigh],dtype=int)
    nlTrZ = np.empty([nats,maxneigh],dtype=int)

    if(mpiON): 
        nl = collect_matrix_from_chunks(nlChunk,nats,natsPerRank,rank,numranks,comm)
        nlTrX = collect_matrix_from_chunks(nlTrChunkX,nats,natsPerRank,rank,numranks,comm)
        nlTrY = collect_matrix_from_chunks(nlTrChunkY,nats,natsPerRank,rank,numranks,comm)
        nlTrZ = collect_matrix_from_chunks(nlTrChunkZ,nats,natsPerRank,rank,numranks,comm)
    else:
        nl = nlChunk
        nlTrX = nlTrChunkX
        nlTrY = nlTrChunkY
        nlTrZ = nlTrChunkZ

    if rank == 0:
        for kk in range(nats):
            print("nl",nl[kk,0:5])
    
    #comm.Allgather(nlChunk,nl)
    #comm.Allgather(nlTrChunkX,nlTrX)
    #comm.Allgather(nlTrChunkY,nlTrY)
    #comm.Allgather(nlTrChunkZ,nlTrZ)

    return(nl,nlTrX,nlTrY,nlTrZ)

#def send_and_receive(dataSend,fromRank,toRank,rank,comm):
#    dataRecv = None
#    if rank == fromRank:
#        comm.Send(dataSend, dest=toRank, tag=77)
#    elif rank == toRank:
#        dataRecv = np.empty((len(dataSend[:,0]), len(dataSend[0,:])), dtype='i')
#        comm.Recv(dataRecv, source=fromRank, tag=77)
#    return dataRecv



