# Pytorch kernels

from sdc_system import *

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

    #We will have approximatly [(4/3)*pi * rcut^3 * atomic density] number of neighbors.
    #A very large atomic density could be 1 atom per (1.0 Ang)^3 = 1 atoms per Ang^3
    volBox = get_volBox(latticeVectors,verb=False)
    density = 1.0
    maxneigh = int(3.14592 * (4.0/3.0) * density * rcut**3)

    #We assume the box is orthogonal
    nx = 1 + int(latticeVectors[0,0]/(2.0*rcut))
    ny = 1 + int(latticeVectors[1,1]/(2.0*rcut))
    nz = 1 + int(latticeVectors[2,2]/(2.0*rcut))
    nBox = nx*ny*nz
    maxInBox = int(density*(2.0*rcut)**3) #Upper bound for the max number of atoms per box
    inbox = np.zeros((nBox,maxInBox),dtype=int)
    inbox[:,:] = -1
    totPerBox = np.zeros((nBox),dtype=int)
    totPerBox[:] = -1
    boxOfI = np.zeros((nats),dtype=int)
    xBox = np.zeros((nBox),dtype=int)
    yBox = np.zeros((nBox),dtype=int)
    zBox = np.zeros((nBox),dtype=int)
    ithFromXYZ = np.zeros((nx,ny,nz),dtype=int)

    minx = np.min(coords[:,0])
    miny = np.min(coords[:,1])
    minz = np.min(coords[:,2])

    smallReal = 0.0
    #Search for the box coordinate and index of every atom

    for i in range(nats):
        #Index every atom respect to the discretized position on the simulation box.
        #tranlation = coords[i,:] - origin !For the general case we need to make sure coords are > 0
        ix =  int((coords[i,0] - minx + smallReal)/(2.0*rcut)) #small box x-index of atom i
        iy =  int((coords[i,1] - miny + smallReal)/(2.0*rcut)) #small box y-index 
        iz =  int((coords[i,2] - minz + smallReal)/(2.0*rcut)) #small box z-index

        if(ix > nx or ix < 0): print("Error in box index"); exit(0)
        if(iy > ny or iy < 0): print("Error in box index"); exit(0) 
        if(iz > nz or iz < 0): print("Error in box index"); exit(0) 

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
        inbox[ith,totPerBox[ith]] = i #Who is in both ith
    print(inbox[boxOfI[0],:],totPerBox[0],boxOfI[0],0)

    #For each atom we will look around to see who are its neighbors

    def get_neighs_of(i,boxOfI,ithFromXYZ,inbox,latticeVectors):    
        nlVect = np.zeros((maxneigh),dtype=int)
        nlTrVectX = np.zeros((maxneigh),dtype=int)
        nlTrVectY = np.zeros((maxneigh),dtype=int)
        nlTrVectZ = np.zeros((maxneigh),dtype=int)
        #print("atom",i)
        cnt = -1
        #Which box it beongs to
        ibox = boxOfI[i]
        #Look inside the box and the neighboring boxes
        for ix in range(-1,2):
            for iy in range(-1,2):
                for iz in range(-1,2):
                    #Get neigh box coordinate
                    jxBox = xBox[ibox] + ix
                    jyBox = yBox[ibox] + iy
                    jzBox = zBox[ibox] + iz
                    tx = 0.0 ; ty = 0.0 ; tz = 0.0
                    if(jxBox < 0):
                        jxBox = nx-1
                        tx = -1
                    elif(jxBox == nx):
                        jxBox = 0
                        tx = 1
                    if(jyBox < 0):
                        jyBox = ny-1
                        ty = -1
                    elif(jyBox == ny):
                        jyBox = 0
                        ty = 1
                    if(jzBox < 0):
                        jzBox = nz-1
                        tz = -1
                    elif(jzBox == nz):
                        jzBox = 0
                        tz = 1
                    
                    #Get the neigh box index
                    jbox = ithFromXYZ[jxBox,jyBox,jzBox]
                    #Now loop over the atoms in the jbox
                    for j in range(totPerBox[jbox]):
                        jj = inbox[jbox,j] #Get atoms in box j
                        translation = tx*latticeVectors[0,:] + ty*latticeVectors[1,:] + tz*latticeVectors[2,:]
                        coordsNeigh = coords[jj,:] + translation
                        distance = np.linalg.norm(coords[i,:] - coordsNeigh)
                        if ((distance < rcut) and (distance > 1.0E-12)):
                        #if (True == True):
                            cnt = cnt + 1
                            nlVect[cnt] = jj # jj is a neighbor of i by some translation
                            nlTrVectX[cnt] = tx
                            nlTrVectY[cnt] = ty
                            nlTrVectZ[cnt] = tz
        nlVect[0] = cnt

        return(nlVect,nlTrVectX,nlTrVectY,nlTrVectZ)

    nlChunk = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkX = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkY = np.empty([natsInRank,maxneigh],dtype=int)
    nlTrChunkZ = np.empty([natsInRank,maxneigh],dtype=int)

    for k in range(natsInRank):
        i = natsPerRank*(rank) + k 
        nlVect,nlTrVectX,nlTrVectY,nlTrVectZ = get_neighs_of(i,boxOfI,ithFromXYZ,inbox,latticeVectors)
        nlChunk[k,:] = nlVect[:] 
        nlTrChunkX[k,:] =  nlTrVectX[:]
        nlTrChunkY[k,:] =  nlTrVectY[:]
        nlTrChunkZ[k,:] =  nlTrVectZ[:]

    nl = np.empty([nats,maxneigh],dtype=int)
    nlTrX = np.empty([nats,maxneigh],dtype=int)
    nlTrY = np.empty([nats,maxneigh],dtype=int)
    nlTrZ = np.empty([nats,maxneigh],dtype=int)

    if(mpiON): 
        nl = collect_matrix_from_chunks(nlChunk,nats,natsPerRank,rank,numranks,comm)
        nlTrX = collect_matrix_from_chunks(nlTrChunkX,nats,natsPerRank,rank,numranks,comm)
        nlTrY = collect_matrix_from_chunks(nlTrChunkY,nats,natsPerRank,rank,numranks,comm)
        nlTrZ = collect_matrix_from_chunks(nlTrChunkZ,nats,natsPerRank,rank,numranks,comm)
    else:
        nl = nlChunk
        nlTrX = nlTrChunkX
        nlTrY = nlTrChunkY
        nlTrZ = nlTrChunkZ

    if rank == 0:
        for kk in range(nats):
            print("nl",nl[kk,0:5])
    
    #comm.Allgather(nlChunk,nl)
    #comm.Allgather(nlTrChunkX,nlTrX)
    #comm.Allgather(nlTrChunkY,nlTrY)
    #comm.Allgather(nlTrChunkZ,nlTrZ)

    return(nl,nlTrX,nlTrY,nlTrZ)
