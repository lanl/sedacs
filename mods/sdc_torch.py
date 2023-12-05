# Pytorch kernels

from sdc_system import *
import torch as tc
import torch.nn.functional as tf

## Neighbor list 
# @brief It will bild a neighbor list using an "all to all" approach
# @param coords System coordinates. coords[7,1]: y-coordinate of atom 7.
# @param latticeVectors. Lattice vectors of the system box. latticeVectors[1,2]: z-coordinate of vector 1.
# @param nl neighbor list type: a simple 2D array indicating the neighbors of each atom.
# @param rank MPI rank
#
def build_nlist_torch(coords,latticeVectors,rcut,rank=0,numranks=1,verb=False):

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
#    nats_left = nats % numranks
#    if (rank < nats_left):
#        natsInRank = natsPerRank + 1
#    else:
#        natsInRank = natsPerRank

    #natsInBuff = max(natsInRank,nats - natsPerRank*(numranks - 1))

    #We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    #A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors,verb=False)
    density = 1.0
    maxneigh = int(3.14592 * (4.0/3.0) * density * rcut**3)
    boxSize = rcut

    #We assume the box is orthogonal
    nx = int(latticeVectors[0,0]/boxSize)
    ny = int(latticeVectors[1,1]/boxSize)
    nz = int(latticeVectors[2,2]/boxSize)
    nBox = nx*ny*nz
    maxInBox = int(density*(boxSize)**3) #Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox,maxInBox),dtype=int)
    inbox[:,:] = -1
    totPerBox = np.zeros((nBox),dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats),dtype=int)
    xBox = np.zeros((nBox),dtype=int)
    yBox = np.zeros((nBox),dtype=int)
    zBox = np.zeros((nBox),dtype=int)
    ithFromXYZ = np.zeros((nx,ny,nz),dtype=int)
    neighbox = np.zeros((nBox,27),dtype=int)

    minx = np.min(coords[:,0])
    miny = np.min(coords[:,1])
    minz = np.min(coords[:,2])

    smallReal = 0.0
    #Search for the box coordinate and index of every atom

    for i in range(nats):
        #Index every atom respect to the discretized position on the simulation box.
        #tranlation = coords[i,:] - origin !For the general case we need to make sure coords are > 0
        ix =  int(coords[i,0]/boxSize) % nx #small box x-index of atom i
        iy =  int(coords[i,1]/boxSize) % ny #small box x-index of atom i
        iz =  int(coords[i,2]/boxSize) % nz #small box x-index of atom i
#        ix =  int((coords[i,0] - minx + smallReal)/(2.0*rcut)) #small box x-index of atom i
#        iy =  int((coords[i,1] - miny + smallReal)/(2.0*rcut)) #small box y-index 
#        iz =  int((coords[i,2] - minz + smallReal)/(2.0*rcut)) #small box z-index

        ith =  ix + iy*nx + iz*nx*ny  #Get small box index
        boxOfI[i] = ith

        #From index to box coordinates
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

    #For each box get a flat list of neighboring boxes (including self)
    for i in range(nBox):
        neighbox[i,0] = i
        j = 1
        for ix in range(-1,2):
            for iy in range(-1,2):
                for iz in range(-1,2):
                    if not (ix == 0 and iy == 0 and iz == 0):
                        #Get neigh box coordinate
                        neighx = xBox[i] + ix
                        neighy = yBox[i] + iy
                        neighz = zBox[i] + iz
                        jxBox = neighx % nx
                        jyBox = neighy % ny
                        jzBox = neighz % nz
                        
                        #Get the neigh box index
                        neighbox[i,j] = ithFromXYZ[jxBox,jyBox,jzBox]
                        j += 1

    def get_boxneighs_of(i,boxOfI,ithFromXYZ,inbox,totPerBox):
        boxneighs = np.array([],dtype=int)
        cnt = 1
        #Which box it beongs to
        ibox = boxOfI[i]
        #Look inside the box and the neighboring boxes
        for j in range(27):
            jbox = neighbox[ibox,j]
            boxneighs = np.concatenate((boxneighs,inbox[jbox][0:totPerBox[jbox]]))
        return(boxneighs)

    #For each atom we will look around to see who are its neighbors

    def get_neighs_of(i,boxOfI,ithFromXYZ,inbox,totPerBox,latticeVectors):    
        #print("atom",i)
        cnt = -1
        #Which box it beongs to
        ibox = boxOfI[i]
        boxneighs = get_boxneighs_of(i,boxOfI,ithFromXYZ,inbox,totPerBox)
        #Look inside the box and the neighboring boxes
        #Now loop over the atoms in the jbox
        dvec = np.zeros((len(boxneighs),3),dtype=np.float64)
        for k in range(3):
            dvec[:,k] = (coords[i,k] - coords[boxneighs[:],k] + latticeVectors[k,k]/2.) % latticeVectors[k,k] \
                - latticeVectors[k,k]/2.
        distance = np.array(np.linalg.norm(dvec,axis=1))
        nlVect = boxneighs[np.where(np.logical_and(distance < rcut,distance > 1.0E-12))[0]]

        cnt = len(nlVect)
        nlVect = np.pad(nlVect,(1,maxneigh-cnt-1),'constant',constant_values=(cnt,0))
        return(nlVect)

    nlChunk = np.empty([natsInRank,maxneigh],dtype=int)

 #   firstIdx = natsPerRank*(rank+1)
 #   for i in range(rank-1):
 #       if (i >= nats_left):
 #           firstIdx -= 1
            
    for k in range(natsInRank):
        i = natsPerRank*(rank) + k 
#        i = firstIdx + k
        nlVect = get_neighs_of(i,boxOfI,ithFromXYZ,inbox,totPerBox,latticeVectors)
        nlChunk[k,:] = nlVect[:] 

    nl = np.empty([nats,maxneigh],dtype=int)

    if(mpiON): 
        nl = collect_matrix_from_chunks(nlChunk,nats,natsPerRank,rank,numranks,comm)
    else:
        nl = nlChunk

    #comm.Allgather(nlChunk,nl)
    #comm.Allgather(nlTrChunkX,nlTrX)
    #comm.Allgather(nlTrChunkY,nlTrY)
    #comm.Allgather(nlTrChunkZ,nlTrZ)

    return(nl)
