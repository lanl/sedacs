# Import the proper engine

import os
import subprocess
import sys

import numpy as np

# from sedacs.proxies.first_level import *
from sedacs.file_io import write_xyz_coordinates

is_engine_up = False


__all__ = ["get_instruction", "send_instruction", "get_hamiltonian_files"]


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
    if fileName is None:
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
def get_hamiltonian_files(engine, coords, atomTypes, symbols, verb):
    # Write coordinates in a file
    dataFileName = engine.path + "/data.dat"
    instrFileName = engine.path + "/instructions.dat"
    # Run the server and keep it running
    if not engine.up:
        cmd = engine.run
        subprocess.Popen(["nohup", cmd], stdout=open("/dev/null", "w"))

    write_xyz_coordinates(dataFileName, coords, atomTypes, symbols)
    # sys.exit(0)
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
            sys.exit(0)
    instructionFile.close()

    print("INSTRUCTION", instr)
    if go:
        hamiltonian = read_matrix(dataFileName)
    else:
        return None  # FIXME: This is not the best way to handle this!

    print(hamiltonian)
    engine.up = True
    return hamiltonian
