""" Main sedacs prototype driver

"""

from sdc_loadMods import *

#Pass arguments from comand line 
args = get_args()

#Initialize sedacs 

np.set_printoptions(threshold=sys.maxsize)

#Initialize sdc parameters
sdc,eng,comm,rank,numranks,sy,hindex,graphNL,nl,nlTrX,nlTrY,nlTrZ = init(args)

sdc.verb=True
print('!!!',     graphNL[0], graphNL.shape)

#Perform a graph-adaptive calculation of the density matrix
#get_adaptiveDM(sdc,eng,comm,rank,numranks,sy,hindex,graphNL)

#test_ffield(sy.coords,sy.types,sy.symbols,sy.latticeVectors,nl,nlTrX,nlTrY,nlTrZ)
do_MD(sy.coords,sy.types,sy.symbols,sy.latticeVectors,nl,nlTrX,nlTrY,nlTrZ,sy.vels,0.01, 10000, 0.0)



