#!/usr/bin/env python3

""" test driver 
This is used to run all the tests within the modules
"""

from sedacs.system import *
from sedacs.periodic_table import *

## For coloring text
class textcolors:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    normal = '\033[0m'

## Function to run the tests 
# @brief General function to run a test
# @param testName Name of the routine to be tested
# @param failedTests List of failed tests
#
def run_my_test(testName,failedTests):
    testFunctionName = "test_"+testName
    passed = False
    exit1 = False
    function = globals()[testFunctionName]
    passed = function(exit1)
    if(not passed): failedTests.append(test)
    return passed

# List of all the name of the routins to be tested
# make a list using system calls based on the grep results 
testNames = ["parameters_to_vectors","ptable","read_xyz_file","write_xyz_coordinates" \
        ,"get_volBox","vectors_to_parameters","read_pdb_file","write_pdb_file"]

print("\nRunning tests ...")

# Run the tests 
exit1 = False
failedTests = []
for test in sorted(testNames):
    run_my_test(test,failedTests)

if(len(failedTests) > 0):
    print(textcolors.red +"\nThe following tests FAILED:" + textcolors.normal)
    for i in range(len(failedTests)):
        print("  -",failedTests[i])
print("\n")
