#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

from sdc_loadMods import *

#Pass arguments from comand line 
args = get_args()

#Initialize sedacs 

np.set_printoptions(threshold=sys.maxsize)
sdc,eng,comm,rank,numranks,sy,hindex,graphNL,nl = init(args)
sdc.verb=True
print('!!!',     graphNL[0], graphNL.shape)
#Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc,eng,comm,rank,numranks,sy,hindex,graphNL)



