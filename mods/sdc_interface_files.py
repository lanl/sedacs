#We need H and other stuff 

#Import the proper engine
from proxy_a import *
global engineUp
engineUp = False
import os
import subprocess
from sdc_system import *
from tempfile import TemporaryFile

## Print a matrix
# @brief Writes a numpy 2D array (a matrix)
# @param mat 2D numpy array
# @param fileName Name of the file to be written
#
def read_matrix(fileName):
    mat = np.load(fileName+".npy",allow_pickle=True)
    open(fileName+".npy","r").close()
    return mat

def get_instruction(fileName):
    instrFile = open(fileName,"r")
    for lines in instrFile:
        instr = lines.split()[0]
    instrFile.close()
    return instr

def send_instruction(instruction,fileName):
    if(fileName == None):
        fileName = "/tmp/actions.act"

    haveFile = os.path.exists(fileName)

    if(not haveFile):
        cmd = "echo NONE > " + fileName
        os.system(cmd)

    #Hold the execution until START is in the file!
    go = False
    while (not go):
        actionFile = open(fileName,"r")
        for lines in actionFile:
            print(lines.split()[0])
            action = lines.split()[0]
        actionFile.close()
        if(action == "START"): go = True

    actionFile = open(fileName,"w")
    print("Action File",fileName,"Instruction:",instruction)
    print(instruction,file=actionFile)
    actionFile.close()


def sdc_get_hamiltonian_files(eng,coords,atomTypes,symbols,verb):
            
    #Write coordinates in a file
    dataFileName = eng.path + "/data.dat"
    instrFileName = eng.path + "/actions.act"
    #Run the server and keep it running
    if(not eng.up):
        cmd = eng.run 
        subprocess.Popen(['nohup', cmd],
            stdout=open('/dev/null', 'w')
            )
   
    write_xyz_coordinates(dataFileName,coords,atomTypes,symbols)
    #exit(0)
    send_instruction("GET_HAMILTONIAN",instrFileName)
   
    instr = get_instruction(instrFileName)
 
    #Hold the execution until START is in the file!
    go = False
    while (not go):
        actionFile = open(instrFileName,"r")
        for lines in actionFile:
            print(lines.split()[0])
            action = lines.split()[0]
        actionFile.close()
        if(action == "START"): go = True
        if(action == "STOP"): exit(0)
    actionFile.close()

    print("INSTRUCTION",instr)
    if(go):
        ham = read_matrix(dataFileName)

    print(ham)

    eng.up = True

    return ham 



