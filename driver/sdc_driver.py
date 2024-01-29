#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

from sdc_loadMods import *

#Pass arguments from comand line 
args = get_args()

#Initialize sedacs 
sdc,eng,comm,rank,numranks,sy,hindex,graphNL,nl = init(args)

#Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc,eng,comm,rank,numranks,sy,hindex,graphNL)



