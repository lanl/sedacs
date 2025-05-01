"""Init proxy
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys
from sedacs.dev.io import src_path
import scipy.linalg as sp
import numpy as np
from proxies.python.proxy_global import read_ppots, bring_ppots, print_ppots, read_tbparams, bring_tbparams 
from proxies.python.proxy_global import bring_ham_list, bring_dm_list, touch_ham_list, touch_dm_list, bring_accel_lib, touch_accel_lib
from proxies.python.proxy_global import bring_cublas_handle_list, bring_stream_list, touch_cublas_handle_list, touch_stream_list
from proxies.python.proxy_global import bring_dev_list, touch_dev_list
import ctypes

__all__ = [
    "init_proxy_proxy","init_proxy_accelerators"
]


## Initialize proxy code 
# @brief We will read all the parameters needed for the 
# guest or proxy code to run. Every guest code will need to 
# set up an initialization function and save parameters that 
# need to be read from file only once. Bond integral parameter, 
# pair potential, etc. will be stored in memory by the guest code.
#
def init_proxy_proxy(symbols,bas_per_atom):
    ...

    # We should really have this be defined in the engine or something, it's hard to follow where
    # these things are intended to be happening.


    #Some codes will have their own input file
    #read_proxy_input_file()
    #Read pair potentials
    read_ppots("ppots.dat",symbols) # Not sure what this is but it probably isn't intended to be hardcoded.
    #print_ppots()
    #Read tb parameters
    filename = "aosa_parameters.dat"
    read_tbparams(filename, symbols, bas_per_atom)

def init_proxy_accelerators(nparts,norbs,rank,full_accel_path):

    try:
        import ctypes
        gpuLib = True
        arch = "nvda"
        pwd = os.getcwd()

        if arch == "nvda":
            print("loading nvidia...")
            lib = ctypes.CDLL("/home/finkeljo/vescratch/sedacs-gpmdsp2/src/sedacs/gpu/nvda/libnvda.so")
        if arch == "amd":
            lib = ctypes.CDLL("/home/finkeljo/vescratch/sedacs-gpmdsp2/src/sedacs/gpu/amd/libamd.so")

        import gpuLibInterface as gpu

    except:
        gpuLib = False

    #set devices based on rank
    num_devices=2   #num devices per node
    gpu.set_device(rank%num_devices,lib)

    touch_accel_lib(full_accel_path)
    accel_lib=bring_accel_lib()

    size_of_double = 8  #bytes
    size_of_float  = 4  #bytes
    size_of_half   = 2  #bytes
    matSize   = norbs * norbs * size_of_double
    matSize_f = norbs * norbs * size_of_float
    matSize_h = norbs * norbs * size_of_half

    # initialize global list storing ham and dm device vars
    touch_cublas_handle_list()
    touch_stream_list()
    touch_dev_list()
   
    cublas_handle_list=bring_cublas_handle_list()
    stream_list=bring_stream_list()
    dev_list=bring_dev_list()

    print("CALLING GPU DEVALLOC!!---------------------")
    for i in range(0,1):
        print("Allocating device mem, handles and streams ...")
        cublas_handle_list.append(gpu.cublasInit(accel_lib))
        stream_list.append(gpu.set_stream(accel_lib))
    

    # if sp2
    for i in range(0,3):
        print("Allocating device mem, handles and streams ...")
        dev_list.append(gpu.dev_alloc(matSize, lib))
    for i in range(3,8):
        print("Allocating device mem, handles and streams ...")
        dev_list.append(gpu.dev_alloc(matSize_f, lib))
    for i in range(8,10):
        print("Allocating device mem, handles and streams ...")
        dev_list.append(gpu.dev_alloc(matSize_h, lib))
    for i in range(10,12):
        print("Allocating pinned mem ...")
        dev_list.append(gpu.host_alloc_pinned(matSize, lib))

