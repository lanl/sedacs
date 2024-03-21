#!/usr/bin/env python3

""" simple test runner
This is used to run individual tests
"""

from sdc_system import *
from sdc_ptable import *

## For coloring text
class textcolors:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    normal = '\033[0m'

## Function to run the tests 
# @brief General function to run a test
# @param testName Name of the routine to be tested
#
def run_my_test(testName):
    testFunctionName = "test_"+testName
    passed = False
    exit1 = True
    function = globals()[testFunctionName]
    passed = function(exit1)

# List of all the name of the routins to be tested
# make a list using system calls based on the grep results 
testName = sys.argv[1] 

print("\nRunning tests ...",testName)

# Run the tests 
run_my_test(testName)

