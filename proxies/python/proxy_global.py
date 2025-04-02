#Set some global variables such 
#as the parameters that are needed 
#throughout the code

import numpy as np
from proxies.python.hamiltonian_elements import AOSA_Parameter
import ctypes

ppots      = None 
tbparams   = None 
d_ham_list = None
d_dm_list  = None

def read_ppots(filename,symbols):
    myfile = open(filename,"r")
    global ppots
    ppots = np.zeros((len(symbols),len(symbols),4))

    for lines in myfile:
        info = lines.split()
        if(len(info) > 1):
            pairI = info[0] 
            pairJ = info[1]
            for i in range(len(symbols)):
                for j in range(len(symbols)):
                    if(symbols[i] == pairI and symbols[j] == pairJ):
                        for k in range(2,len(info)):
                            ppots[i,j,k-2] = float(info[k])
                    ppots[j,i,:] =  ppots[i,j,:]



def print_ppots():
    print(ppots)

def bring_ppots():
    return ppots

def bring_cublas_handle_list():
    return cublas_handle_list

def bring_stream_list():
    return stream_list

def bring_ham_list():
    return d_ham_list

def bring_dm_list():
    return d_dm_list

def bring_dev_list():
    return dev_list

def touch_cublas_handle_list():
    global cublas_handle_list
    cublas_handle_list=[]

def touch_stream_list():
    global stream_list
    stream_list=[]

def touch_ham_list():
    global d_ham_list
    d_ham_list=[]

def touch_dm_list():
    global d_dm_list
    d_dm_list=[]

def touch_dev_list():
    global dev_list
    dev_list=[]

def touch_accel_lib(full_path):
    global accel_lib
    #lib = ctypes.CDLL("/home/finkeljo/sedacs-gpmdsp2/src/sedacs/gpu/nvda/libnvda.so")
    accel_lib = ctypes.CDLL(str(full_path)+"/libnvda.so")

def bring_accel_lib():
    return accel_lib

def read_tbparams(filename,symbols,bas_per_atom):

    ntypes = len(symbols)

    global tbparams

    tbparams = [None]*ntypes
    sOrbTypes = ["s"]
    spOrbTypes = ["s","px","py","pz"]

    for i in range(ntypes):
        sublist = [None]*bas_per_atom[i]
        for j in range(bas_per_atom[i]):
            if(bas_per_atom[i] == 1):
                orbTypes = sOrbTypes
            elif(bas_per_atom[i] == 4):
                orbTypes = spOrbTypes
            param = AOSA_Parameter(symbols[i],orbTypes[j],filename)
            sublist[j] = param 
        tbparams[i] = sublist


def bring_tbparams():
    return tbparams

