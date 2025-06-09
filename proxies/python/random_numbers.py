"""random number generator
This code is only used to guide implemetations and understand which are the 
basic elements needed to interface with the sedacs driver.
"""

import os
import sys

__all__ = [
    "RandomNumberGenerator",
]


## Simple random number generator
# This is important in order to compare across codes
# written in different languages.
#
# To initialize:
# \verbatim
#   myRand = rand(123)
# \endverbatim
# where the argument of rand is the seed.
#
# To get a random number between "low" and "high":
# \verbatim
#   rnd = myRand.get_rand(low,high)
# \endverbatim
#
class RandomNumberGenerator:
    """To generate random numbers."""

    def __init__(self, seed):
        self.a = 321
        self.b = 231
        self.c = 13
        self.seed = seed
        self.status = seed * 1000

    def generate(self, low, high):
        """Get a random real number in between low and high."""
        w = high - low
        place = self.a * self.status
        place = int(place / self.b)
        rand = (place % self.c) / self.c
        place = int(rand * 1000000)
        self.status = place
        rand = low + w * rand

        return rand


if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("Give the total number of vector elements. Example:\n")
        print("python random.py 100\n")
        sys.exit(0)
    else:
        nel = int(sys.argv[1])

    verb = True

    rnd = RandomNumberGenerator(1234)

    for i in range(nel):
        print(i,rnd.generate(0,1))



