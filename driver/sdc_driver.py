#!/usr/bin/env python3
""" Main sedacs prototype driver

"""

from sdc_loadMods import *
import sys
#Pass arguments from comand line 
args = get_args()

#Initialize sedacs 
sdc,eng,comm,rank,numranks,sy,hindex,graphNL,nl = init(args)

#Stops at Init if 
if sdc.stopAt=="Init":  
    sys.exit()

#Perform a graph-adaptive calculation of the density matrix
get_adaptiveDM(sdc,eng,comm,rank,numranks,sy,hindex,graphNL)



