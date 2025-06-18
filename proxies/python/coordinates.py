"""coordinates
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys
import numpy as np
from random_numbers import RandomNumberGenerator

__all__ = [
    "get_random_coordinates",
]


## Generating random coordinates
# @brief Creates a system of size "nats = Number of atoms" with coordindates having
# a random (-1,1) displacement from a simple cubic lattice with parameter 2.0 Ang.
# This funtion is only used for testing purposes.
# @param nats The total number of atoms
# @return coordinates Position for every atom. z-coordinate of atom 1 = coords[0,2]
#
def get_random_coordinates(nats):
    """Get random coordinates"""
    length = int(nats ** (1 / 3)) + 1
    coords = np.zeros((nats, 3))
    latticeParam = 2.0
    atomsCounter = -1
    myrand = RandomNumberGenerator(111)
    for i in range(length):
        for j in range(length):
            for k in range(length):
                atomsCounter = atomsCounter + 1
                if atomsCounter >= nats:
                    break
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 0] = i * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 1] = j * latticeParam + rnd
                rnd = myrand.generate(-1.0, 1.0)
                coords[atomsCounter, 2] = k * latticeParam + rnd
    return coords


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the total number of atoms. Example:\n")
        print("python coordinates.py 100\n")
        sys.exit(0)
    else:
        nats = int(sys.argv[1])

    verb = True

    coords = get_random_coordinates(nats)

    print("Coordinates:",coords)
