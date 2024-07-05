# Import the proper engine
from proxy_a import *

global engineUp
engineUp = False
import os
import subprocess
from sedacs.system import System
from tempfile import TemporaryFile


## Write a matrix
# @brief Writes a numpy 2D array (a matrix)
# @param fileName Name of the file to be written
# @return mat 2D numpy array
#
def read_matrix(fileName):
    mat = np.load(fileName + ".npy", allow_pickle=True)
    open(fileName + ".npy", "r").close()
    return mat


## Read instruction
# @brief Reads an instruction from the instruction file
# @param fileName Name of the instructions file
# @return instr String containing instruction
#
def get_instruction(fileName):
    instrFile = open(fileName, "r")
    for lines in instrFile:
        instr = lines.split()[0]
    instrFile.close()
    return instr


## Send instruction
# @brief This will write an instruction into the instruction file
# @param fileName The name of the instruction file
#
def send_instruction(instruction, fileName):
    if fileName == None:
        fileName = "/tmp/instructions.dat"

    haveFile = os.path.exists(fileName)

    if not haveFile:
        cmd = "echo NONE > " + fileName
        os.system(cmd)

    # Hold the execution until START is in the file!
    go = False
    while not go:
        instructionFile = open(fileName, "r")
        for lines in instructionFile:
            print(lines.split()[0])
            instruction = lines.split()[0]
        instructionFile.close()
        if instruction == "START":
            go = True

    instructionFile = open(fileName, "w")
    print("Action File", fileName, "Instruction:", instruction)
    print(instruction, file=instructionFile)
    instructionFile.close()


## Get Hamiltonian
# @brief Get a Hamiltonian using a file type of interface
# @param eng Engine object containing the description of the engine
# @param A 2D Nx3 numpy array that stores the position for every atom.
# Example: z-coordinate of atom 1 = `coords[0,2]`. It can be initialized
# as `coords = np.zeros((nats,3))` where `nats` is the number of atoms.
# @param atomTypes for each atom, e.g., the first atom is of type `atomTypes[0]`. This can be initialized as `atomTypes = np.zeros((nats),dtype=int)`
# @param symbols Symbols for each atom type, e.g, the element symbol of the first atom is `symbols[types[0]]`
# @param verb Verbosity level
#
def sdc_get_hamiltonian_files(eng, coords, atomTypes, symbols, verb):
    # Write coordinates in a file
    dataFileName = eng.path + "/data.dat"
    instrFileName = eng.path + "/instructions.dat"
    # Run the server and keep it running
    if not eng.up:
        cmd = eng.run
        subprocess.Popen(["nohup", cmd], stdout=open("/dev/null", "w"))

    write_xyz_coordinates(dataFileName, coords, atomTypes, symbols)
    # exit(0)
    send_instruction("GET_HAMILTONIAN", instrFileName)

    instr = get_instruction(instrFileName)

    # Hold the execution until START is in the file!
    go = False
    while not go:
        instructionFile = open(instrFileName, "r")
        for lines in instructionFile:
            print(lines.split()[0])
            instruction = lines.split()[0]
        instructionFile.close()
        if instruction == "START":
            go = True
        if instruction == "STOP":
            exit(0)
    instructionFile.close()

    print("INSTRUCTION", instr)
    if go:
        ham = read_matrix(dataFileName)

    print(ham)

    eng.up = True

    return ham
