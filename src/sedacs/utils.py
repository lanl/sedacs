"""utils
Some auxiliary functions that might be used in several modules

"""

import numpy as np


## Compare to files
# @brief Compare two files and report True if they are equal
# @param fileName1 Name of the first file
# @param fileName2 Name of the second file
# @return filesAreEqual True or False depending if the files are equal or not
#
def sdc_files_are_equal(fileName1, fileName2):
    file1 = open(fileName1, "r")
    file2 = open(fileName2, "r")

    filesAreEqual = True
    for lines1 in file1:
        lines2 = file2.readline()
        lines1Stp = lines1.strip()
        lines2Stp = lines2.strip()
        if lines1Stp != lines2Stp:
            filesAreEqual = False
            break

    return filesAreEqual
