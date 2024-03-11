"""system
Some functions to create, read, and manupulate coordinates of a chemical system

So far: Creates random coordinates; reads and writes xyz and pdb files;
creates a neighbor list.
"""
import numpy as np

global aseLib
try: import ase1.io; aseLib = True
except: aseLib = False

global mdtrajLib
try: import mdtraj as md; mdtrajLib=True
except: mdtrajLib=False

from sdc_ptable import ptable

#from sdc_out import *
try:
    from mpi4py import MPI
    mpiLib = True
except ImportError as e:
    mpiLib = False
from multiprocessing import Pool
if(mpiLib): 
    from sdc_mpi import * 
import time

## Chemical system type 
# @brief To be used only when really needed! 
# 
class system:
    """A prototype for the system type.
    """
    def __init__(self,nats=1):
        ## Number of atoms
        self.nats = nats
        ## Number of core atoms 
        self.ncores = self.nats 
        ## Number of atom types 
        self.ntypes = nats
        ## Type for each atom, e.g., the first atom is of type "types[0]"
        self.types = np.zeros(self.nats,dtype=int) 
        ## Coordinates for each atom, e.g., z-coordinate of the frist atom is coords[0,2]
        self.coords = np.zeros((self.nats,3),dtype=float)
        ## LatticeVectors. 3x3 matrix containing the lattice vectors for the simulation box.
        # latticeVectors[1,:] = first lattice vector.
        self.latticeVectors = np.zeros((3,3),dtype=float)
        ## Symbols for each atom type, e.g, the element symbol of the first atom is symbols[types[0]]
        self.symbols = ptable().symbols[self.types] 
        ## Number of atomic orbital for each type
        self.orbs = np.ones(self.nats,dtype=int)

    def print_summary(self):
        s = """nats = {nats}
ncores = {ncores}
ntypes = {ntypes}
coords[0] = {coords}
latticeVectors[0] = {latticeVectors}
symbols = {symbols}
orbs = {orbs}"""        
        print(s.format(nats=self.nats,ncores=self.ncores,ntypes=self.ntypes, \
                        coords=self.coords[0], \
                        latticeVectors=self.latticeVectors[0], \
                        symbols=self.symbols, \
                        orbs=self.orbs))
    if mdtrajLib:
        def from_mdtraj(self,traj,frame_idx=0):
            table,bonds = traj.topology.to_dataframe()
            self.symbols = table['element'].to_numpy().tolist()
            self.nats = len(self.symbols)
            self.ncores = self.nats
            self.ntypes = self.nats
            #multiply the following two by 10. to convert to Angstroms
            self.coords = 10.*traj.xyz[frame_idx].astype(float)
            if traj.unitcell_vectors is not None:
                self.latticeVectors = 10.*traj.unitcell_vectors[frame_idx].astype(float)
            else:
                import warnings
                warnings.warn("No unit cell information in this mdtraj trajectory. If unit cell information is desired, it can be obtained by loading a .pdb file as a trajectory.")
            self.orbs = np.ones(self.nats,dtype=int)
            
## Trajectory type 
# @brief To handle simulation results
# @param system The system description with topology info
# @param coords Coordinates at each snapshot (Angstrom)
# @param latticeVectors Simulation box vectors (Angstrom)
# @param value Generic atom value (e.g. electron population) at each time point
# @param timestep Time difference between frames, for uniform sampling (ps)
# @param time Time at each frame (ps)

class trajectory:
    """A prototype for the trajectory type.
    """
    
    def __init__(self,sys=None,nats=1,nframes=1,timestep=0.00025):
        if sys is None:
            self.system = system(nats)
        else:
            self.system = sys
            nats=sys.nats
        self.coords = np.zeros((nframes,nats,3),dtype=float)
        self.latticeVectors = None
        self.value = None
        self.timestep = timestep
        self.time = np.ones(nframes,dtype=float)*self.timestep

    def from_mdtraj(self,traj):
        self.system = system()
        self.system.from_mdtraj(traj)
        self.coords = 10.*traj.xyz.astype(float)
        if traj.unitcell_vectors is not None:
            self.latticeVectors = 10.*traj.unitcell_vectors.astype(float)
        self.time = traj.time.astype(float)
        if traj.n_frames >=2:
            self.timestep = traj.timestep

    def slice(self,first=0,last=None,skip=1):
        if last is None:
            last = len(self.coords)
        self.coords = self.coords[first:last+1:skip]
        self.time = self.time[first:last+1:skip]
        if self.value is not None:
            self.value = self.value[first:last+1:skip]
        if self.latticeVectors is not None:
            self.latticeVectors = self.latticeVectors[first:last+1:skip]
        self.timestep = self.timestep * skip
            

    def load_prg_xyz(self,fname):
        with open(fname) as f:
            lines = np.array(f.readlines())
            nats = int(lines[0])
            if nats != self.system.nats:
                raise Exception("Number of atoms must be same as that in system")
            mask = np.ones(len(lines),dtype=bool)
            mask[np.arange(0,len(lines),nats+2)] = False
            mask[np.arange(1,len(lines),nats+2)] = False
            lines = lines[mask]
            xyzc = np.loadtxt(lines.tolist(),usecols=range(1,5)).astype(float)
            nframes = int(len(xyzc)/nats)
            xyzc = np.reshape(xyzc,(nframes,nats,4))
            self.coords = xyzc[:,:,0:3]
            self.value = xyzc[:,:,3:4]

    if mdtrajLib:
        def save_xtc(self,fname):
            from mdtraj.formats import XTCTrajectoryFile
            with XTCTrajectoryFile(fname, 'w') as f:
                if self.latticeVectors is not None:
                    f.write(self.coords/10.,box=self.latticeVectors/10.)
                else:
                    f.write(self.coords/10.,box=np.repeat(self.system.latticeVectors[np.newaxis,:,:]/10.,len(self.coords),axis=0))

        def save_dcd(self,fname):
            from mdtraj.formats import DCDTrajectoryFile
            with DCDTrajectoryFile(fname, 'w') as f:
                if self.latticeVectors is not None:
                    latticeVectors = self.latticeVectors
                else:
                    latticeVectors=np.repeat(self.system.latticeVectors[np.newaxis,:,:]/10.,len(self.coords),axis=0)                    
                latticeParams=vectors_to_parameters(latticeVectors)
                f.write(self.coords,cell_lengths= \
                        latticeParams[:,0:3], \
                        cell_angles=latticeParams[:,3:6])

        def save_netcdf(self,fname):
            from mdtraj.formats import NetCDFTrajectoryFile
            with NetCDFTrajectoryFile(fname, 'w') as f:
                if self.latticeVectors is not None:
                    latticeVectors=self.latticeVectors
                else:
                    latticeVectors=np.repeat(self.system.latticeVectors[np.newaxis,:,:]/10.,len(self.coords),axis=0)
                latticeParams=vectors_to_parameters(latticeVectors)
                f.write(self.coords,cell_lengths= \
                        latticeParams[:,0:3], \
                        cell_angles=latticeParams[:,3:6])
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

    #pi = 3.1415926535897932384626433832795
    pi = np.pi

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

## Transforms the lattice vectors to lattice parameters
# @param latticeVectors 3x3 array containing the lattice vectors
# @param verb Verbosity level.
#
def vectors_to_parameters(Amat,verb=False):
    if Amat.ndim == 3:
        a = np.sqrt(np.einsum('ij,ij->i',Amat[:,0],Amat[:,0]))
        b = np.sqrt(np.einsum('ij,ij->i',Amat[:,1],Amat[:,1]))
        c = np.sqrt(np.einsum('ij,ij->i',Amat[:,2],Amat[:,2]))
        adotb = np.einsum('ij,ij->i',Amat[:,0],Amat[:,1])
        adotc = np.einsum('ij,ij->i',Amat[:,0],Amat[:,2])
        bdotc = np.einsum('ij,ij->i',Amat[:,1],Amat[:,2])
        alpha = np.arccos(bdotc/b/c)*180./np.pi
        beta = np.arccos(adotc/a/c)*180./np.pi
        gamma = np.arccos(adotb/a/b)*180./np.pi
        alpha[np.abs(alpha-90.) <= 1.e-5] = 90.
        beta[np.abs(alpha-90.) <= 1.e-5] = 90.
        gamma[np.abs(alpha-90.) <= 1.e-5] = 90.
    else:
        a = np.sqrt(np.inner(Amat[0],Amat[0]))
        b = np.sqrt(np.inner(Amat[1],Amat[1]))
        c = np.sqrt(np.inner(Amat[2],Amat[2]))
        adotb = np.inner(Amat[0],Amat[1])
        adotc = np.inner(Amat[0],Amat[2])
        bdotc = np.inner(Amat[1],Amat[2])
        alpha = np.arccos(bdotc/b/c)*180./np.pi
        beta = np.arccos(adotc/a/c)*180./np.pi
        gamma = np.arccos(adotb/a/b)*180./np.pi
        if abs(alpha-90.) <= 1.e-5:
            alpha = 90.
        if abs(beta-90.) <= 1.e-5:
            beta = 90.
        if abs(gamma-90.) <= 1.e-5:
            gamma = 90.
    return np.transpose(np.array((a,b,c,alpha,beta,gamma)))

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


## Coordinates main reader
# @brief This will read the coodinates of a chemical system (so far only xyz and pdb
# are available).

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
    print("CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1 ",file=myFileOut) # $$$ ??? Maybe adaptive vectors ???
    print("MODEL",file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print("ATOM",'{:6d}'.format(i+1)," "+symb,"  MOL",'{:5d}'.format(molIds[i]),"    ", '{:05.3f}'.format(coords[i,0]),"", '{:05.3f}'.format(coords[i,1]), \
               "",'{:05.3f}'.format(coords[i,2])," 1.00  0.00          ",symb,file=myFileOut)
    print("TER",file=myFileOut)
    print("END",file=myFileOut)


## Write coordinates into an xyz file
# 
# @param fileName File name 
# @param coords Position for every atoms. z-coordinate of atom 1 = coords[0,2]
# @param types list of types for every atom in the system. 
# @param symbols Symbols for every atom type
# 
def write_xyz_coordinates(fileName,coords,types,symbols):
    """Writes coordinates in simple pdb format
    """
    nats = len(coords[:,1])
    myFileOut = open(fileName,"w")
    print(nats,file=myFileOut)
    print("xyz format",file=myFileOut)
    for i in range(nats):
        symb = symbols[types[i]]
        print(symb,coords[i,0],coords[i,1],coords[i,2],file=myFileOut)

    myFileOut.close()

## Extract subsystem
# @brief Extracs a chemical subsystem (coordinates and atomic types) 
# from a larger system using a set of indices. 
# @param coords Position for every atom. z-coordinate of atom 1 = coords[0,2]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @param symbols Symbols for every atom type
# @param part list of index for the part to be extracted
# @return subSyCoords Subsystem atomic coordinates
# @return subSyTypes Subsystem atomic types
#
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
    maxx = np.max(coords[:,0])
    maxy = np.max(coords[:,1])
    maxz = np.max(coords[:,2])
    minx = np.min(coords[:,0])
    miny = np.min(coords[:,1])
    minz = np.min(coords[:,2])
    
    smallReal = 0.1 #To ensure the borders are contained in the limiting boxes

    #This part is for trying integer discretization of the coordinates
    dr = 0.1 #Discretization param 
    cx = np.zeros((nats),dtype=int)
    cy = np.zeros((nats),dtype=int)
    cz = np.zeros((nats),dtype=int)
    lx = latticeVectors[0,0]/dr
    ly = latticeVectors[1,1]/dr
    lz = latticeVectors[2,2]/dr
    for i in range(nats):
        cx[i] = int(coords[i,0]/dr)
        cy[i] = int(coords[i,1]/dr)
        cz[i] = int(coords[i,2]/dr)

    nx =  int((maxx - minx)/(rcut))
    ny =  int((maxy - miny)/(rcut))
    nz =  int((maxz - minz)/(rcut))
    dx = (maxx - minx + smallReal)/float(nx)
    dy = (maxy - miny + smallReal)/float(ny) 
    dz = (maxz - minz + smallReal)/float(nz)

    ix =  int((maxx - minx + smallReal)/(dx)) #small box x-index of atom i
    iy =  int((maxy - miny + smallReal)/(dy)) #small box y-index 
    iz =  int((maxz - minz + smallReal)/(dz)) #small box z-index

    nBox = nx*ny*nz
    maxInBox = int(density*(rcut)**3) #Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox,maxInBox),dtype=int)
    inbox[:,:] = -1
    totPerBox = np.zeros((nBox),dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats),dtype=int)
    xBox = np.zeros((nBox),dtype=int)
    yBox = np.zeros((nBox),dtype=int)
    zBox = np.zeros((nBox),dtype=int)
    ithFromXYZ = np.zeros((nx,ny,nz),dtype=int)

    #Search for the box coordinate and index of every atom
    for i in range(nats):
        #Index every atom respect to the discretized position on the simulation box.
        #tranlation = coords[i,:] - origin !For the general case we need to make sure coords are > 0
        ix =  int((coords[i,0] - minx )/(dx)) #small box x-index of atom i
        iy =  int((coords[i,1] - miny )/(dy)) #small box y-index 
        iz =  int((coords[i,2] - minz )/(dz)) #small box z-index

        if(ix > nx or ix < 0): print("Error in box index"); exit(0)
        if(iy > ny or iy < 0): print("Error in box index"); exit(0) 
        if(iz > nz or iz < 0): print("Error in box index"); exit(0) 

        ith =  ix + iy*nx + iz*nx*ny  #Get small box index
        boxOfI[i] = ith

        #From index to box coordinates
        #print("ith",ith,nBox,ix,iy,iz,nx,ny)
        xBox[ith] = ix
        yBox[ith] = iy
        zBox[ith] = iz

        #From box coordinates to index  
        ithFromXYZ[ix,iy,iz] = ith

        totPerBox[ith] = totPerBox[ith] + 1 #How many per box
        if(totPerBox[ith] >= maxInBox): 
            print("Exceeding the max in box allowed")
            exit(0)
        inbox[ith,totPerBox[ith]] = i #Who is in box ith

    for i in range(nBox): #Correcting - from indexing to 
        totPerBox[i] = totPerBox[i] + 1

    rcut2 = rcut*rcut

    #For each atom we will look around to see who are its neighbors
    def get_neighs_of(i,boxOfI,ithFromXYZ,inbox,latticeVectors):    
        nlVect = np.zeros((maxneigh),dtype=int)
        nlTrVectX = np.zeros((maxneigh),dtype=int)
        nlTrVectY = np.zeros((maxneigh),dtype=int)
        nlTrVectZ = np.zeros((maxneigh),dtype=int)
        translation = np.zeros((3))
        cnt = 0
        #Which box it beongs to
        ibox = boxOfI[i]
        #Look inside the box and the neighboring boxes
        xBoxIbox = xBox[ibox] ; yBoxIbox = yBox[ibox] ; zBoxIbox = zBox[ibox]
        for ix in range(-1,2):
            for iy in range(-1,2):
                for iz in range(-1,2):
                    #Get neigh box coordinate
                    jxBox = xBoxIbox + ix
                    jyBox = yBoxIbox + iy
                    jzBox = zBoxIbox + iz
                    tx = 0.0 ; ty = 0.0 ; tz = 0.0 ; tr = False
                    if(jxBox < 0):
                        jxBox = nx-1
                        tx = -1
                        tr = True
                    elif(jxBox == nx):
                        jxBox = 0
                        tx = 1
                        tr = True
                    if(jyBox < 0):
                        jyBox = ny-1
                        ty = -1
                        tr = True
                    elif(jyBox == ny):
                        jyBox = 0
                        ty = 1
                        tr = True
                    if(jzBox < 0):
                        jzBox = nz-1
                        tz = -1
                        tr = True
                    elif(jzBox == nz):
                        jzBox = 0
                        tz = 1
                        tr = True
                    
                    
                    #Get the neigh box index
                    jbox = ithFromXYZ[jxBox,jyBox,jzBox]
                    #if (tr):
                    #    translation = tx*latticeVectors[0,:] + ty*latticeVectors[1,:] + tz*latticeVectors[2,:]
                    #else:
                    #    translation[:] = 0.0
                    
                    trlx = tx*lx 
                    trly = ty*ly
                    trlz = tz*lz 
                    #Now loop over the atoms in the jbox
                    for j in range(totPerBox[jbox]):
                        jj = inbox[jbox,j] #Get atoms in box j
                        if(tr):
                         #   coordsNeigh = coords[jj,:] + translation
                            cnx = cx[jj] + trlx
                            cny = cy[jj] + trly     
                            cnz = cz[jj] + trlz     
                        else:
                        #    coordsNeigh = coords[jj,:] 
                            cnx = cx[jj] ; cny = cy[jj] ; cnz = cz[jj]

                        #distance = (coords[i,0] - coordsNeigh[0])**2 + \
                        #        (coords[i,1] - coordsNeigh[1])**2 + \
                        #        (coords[i,2] - coordsNeigh[2])**2

                        distance = float((cx[i] - cnx)**2 + (cy[i] - cny)**2 \
                                + (cz[i] - cnz)**2) * dr**2
                        
                        if ((distance < rcut2) and (distance > 1.0E-12)):
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

    return(nl,nlTrX,nlTrY,nlTrZ)


## Get hindex
# @brief hindex will give the orbital index for each atom 
# in the system. 
# The orbital indices for orbital i goes from `hindex[i]` to `hindex[i+1]-1`
# @param orbs A dictionary that give the total orbitals (basis set size) 
# for each atomic type.
# @param symbols Symbol for each atom type. Symbol for first atom type = symbols[0]
# @param types Index type for each atom in the system. Type for first atom = type[0]
# @return norbs Total number of orbitals
# @return hindex Orbital index for each atom in the system
#
def get_hindex(orbs,symbols,types,verb=False):

    nats = len(types[:])
    hindex = np.zeros((nats+1),dtype=int)
    norbs = 0
    for i in range(nats):
        hindex[i] = norbs
        norbs = norbs + orbs[symbols[types[i]]]
        if(verb):
            print("index,type,symb,orb",i,types[i],symbols[types[i]],orbs[symbols[types[i]]])

    hindex[nats] = norbs

    return norbs, hindex
