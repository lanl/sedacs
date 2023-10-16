"""@package coordinates
Some functions to create random coordinates
 
So far just random coordinates.
"""
import numpy as np


class rand:
    """To generate random numbers.
    """
    def __init__(self,seed):
        self.a = 475
        self.b = 38
        self.c = 41
        self.seed = seed
        self.status = seed

    def get_rand(self,low,high):
        """Get a random real number in betwee low and high."""
        w = high - low
        place = self.a*self.status
        place = place/self.b
        rand = (place%self.c)/self.c
        place = rand*100000
        self.status = place
        rand = low + w*rand

        return(rand)


def get_random_coordinates(length):
    """Get random coordinates real number in betwee low and high."""
    nats = length**3
    coords = np.zeros((nats,3))
    latticeParam = 2.0 
    atomsCounter = -1
    myrand = rand(123)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,0] = i*latticeParam + rnd
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,1] = j*latticeParam + rnd
                rnd = myrand.get_rand(-1.0,1.0)
                coords[atomsCounter,2] = k*latticeParam + rnd 
    return coords

def write_xyz_coordinates(coords,symbols):
    nats = len(coords[:,1])
    myFileOut = open("coords.xyz","w")
    print(nats,file=myFileOut)
    print("coords.xyz",file=myFileOut)
    for i in range(nats):
        print(symbols[i],coords[i,0],coords[i,1],coords[i,2],file=myFileOut)



